<#
.SYNOPSIS
    Starts the JMP Docs Assistant chat app, and makes sure Ollama is up first.

.DESCRIPTION
    Run this instead of typing the streamlit command by hand. It:
      - starts the Ollama server if it is not already listening
      - starts Streamlit detached, so it keeps running after you close the window
      - waits until the app actually answers, rather than assuming it worked
      - health-checks rather than just port-checks, and replaces a hung copy
      - tells you up front if the search index has not been built yet

.EXAMPLE
    .\Start-JmpDocs.ps1
    .\Start-JmpDocs.ps1 -Stop
    .\Start-JmpDocs.ps1 -Status
    .\Start-JmpDocs.ps1 -Restart
#>
[CmdletBinding()]
param(
    [int]$Port = 8501,
    [switch]$Stop,
    [switch]$Status,
    [switch]$Restart
)

$ErrorActionPreference = 'Stop'

$Proj      = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python    = Join-Path $Proj '.venv\Scripts\python.exe'
$OllamaExe = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
$Log       = Join-Path $env:TEMP 'jmpdocs-streamlit.log'
$PidFile   = Join-Path $env:TEMP 'jmpdocs-streamlit.pid'

# Resolve the data directory the same way the app does: .env wins over the
# config.yaml default of ./data.
$DataDir = Join-Path $Proj 'data'
$EnvFile = Join-Path $Proj '.env'
if (Test-Path $EnvFile) {
    $line = Select-String -Path $EnvFile -Pattern '^\s*JMPDOCS_PATHS__DATA_DIR\s*=\s*(.+)$' |
            Select-Object -First 1
    if ($line) { $DataDir = $line.Matches[0].Groups[1].Value.Trim().Trim('"') }
}
$IndexFile = Join-Path (Join-Path $DataDir 'index') 'faiss.index'

function Test-Port([int]$p) {
    [bool](Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue)
}

function Test-AppHealthy([int]$p) {
    # A listening port is NOT proof the app works. A process can hold 8501 while
    # serving an error page. Ask Streamlit's health endpoint instead.
    try {
        return (Invoke-WebRequest "http://localhost:$p/_stcore/health" `
                    -TimeoutSec 4 -UseBasicParsing).Content -eq 'ok'
    } catch { return $false }
}

function Stop-OnPort([int]$p) {
    foreach ($c in @(Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue)) {
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    # Streamlit runs as a parent + child pair; sweep any strays from this repo.
    Get-CimInstance Win32_Process -Filter "Name like '%python%'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*streamlit*jmpdocs*" -or $_.CommandLine -like "*streamlit run src/jmpdocs*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
}

function Get-AppPid {
    if (-not (Test-Path $PidFile)) { return $null }
    $id = Get-Content $PidFile -ErrorAction SilentlyContinue
    if (-not $id) { return $null }
    $proc = Get-Process -Id $id -ErrorAction SilentlyContinue
    if ($proc) { return [int]$id }
    return $null
}

# ---------------------------------------------------------------- status ----
if ($Status) {
    "ollama (11434) : $(if (Test-Port 11434) { 'running' } else { 'stopped' })"
    if (Test-Port $Port) {
        $healthy = Test-AppHealthy $Port
        "app    ($Port) : listening, $(if ($healthy) { 'healthy' } else { 'NOT RESPONDING - run with -Restart' })"
    } else {
        "app    ($Port) : stopped"
    }
    $appPid = Get-AppPid
    if ($appPid) { "app pid        : $appPid" }
    "index          : $(if (Test-Path $IndexFile) { 'built' } else { 'NOT BUILT - run scripts/build_index.py' })"
    "url            : http://localhost:$Port"
    "log            : $Log"
    return
}

# ------------------------------------------------------------------ stop ----
if ($Stop) {
    $was = (Test-Port $Port) -or (Get-AppPid)
    $appPid = Get-AppPid
    if ($appPid) { Stop-Process -Id $appPid -Force -ErrorAction SilentlyContinue }
    Remove-Item $PidFile -ErrorAction SilentlyContinue
    Stop-OnPort $Port
    if ($was) { "stopped" } else { "nothing was running on port $Port" }
    return
}

# ----------------------------------------------------------------- start ----
if (-not (Test-Path $Python)) {
    throw "virtualenv not found at $Python. Run:  python -m venv .venv ;  .\.venv\Scripts\python.exe -m pip install -e `".[dev]`""
}

# The index must exist, or the app starts fine and then reports that it has
# nothing to search -- which looks like a launcher failure but isn't.
if (-not (Test-Path $IndexFile)) {
    Write-Warning "The search index is missing: $IndexFile"
    Write-Host   "Build it first (about 70 minutes, one time):"
    Write-Host   "    .\.venv\Scripts\python.exe scripts\build_corpus.py"
    Write-Host   "    .\.venv\Scripts\python.exe scripts\build_index.py"
    Write-Host   ""
    Write-Host   "If you built it elsewhere, point .env at it with JMPDOCS_PATHS__DATA_DIR."
    exit 1
}

if (Test-Port $Port) {
    if (-not $Restart -and (Test-AppHealthy $Port)) {
        "app already running and healthy -> http://localhost:$Port"
        return
    }
    # Either the caller asked for a restart, or something is squatting on the
    # port without serving. A port check alone would wrongly report success here.
    if ($Restart) { Write-Host 'restarting...' }
    else { Write-Warning "something is on port $Port but not responding - replacing it" }
    Stop-OnPort $Port
    Remove-Item $PidFile -ErrorAction SilentlyContinue
}

# Ollama must be up or every answer fails with a connection error.
if (-not (Test-Port 11434)) {
    if (-not (Test-Path $OllamaExe)) { throw "Ollama not found at $OllamaExe" }
    Write-Host 'starting ollama...'
    Start-Process -FilePath $OllamaExe -ArgumentList 'serve' -WindowStyle Hidden | Out-Null
    for ($i = 0; $i -lt 30 -and -not (Test-Port 11434); $i++) { Start-Sleep -Seconds 1 }
    if (-not (Test-Port 11434)) { throw 'ollama failed to start' }
}
Write-Host 'ollama ready.'

Remove-Item $Log, "$Log.err" -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath $Python `
    -ArgumentList '-m', 'streamlit', 'run', 'src/jmpdocs/app.py',
                  '--server.headless=true', "--server.port=$Port" `
    -WorkingDirectory $Proj -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $Log -RedirectStandardError "$Log.err"

Set-Content -Path $PidFile -Value $proc.Id -Encoding ascii
Write-Host "starting app (pid $($proc.Id))..."

# Wait for a real response rather than reporting success optimistically.
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 1
    if ($proc.HasExited) { break }
    try {
        if ((Invoke-WebRequest "http://localhost:$Port/_stcore/health" -TimeoutSec 3 -UseBasicParsing).Content -eq 'ok') {
            $ok = $true; break
        }
    } catch { }
}

if ($ok) {
    Write-Host ''
    Write-Host "  JMP Docs Assistant is running" -ForegroundColor Green
    Write-Host "  http://localhost:$Port" -ForegroundColor Green
    Write-Host ''
    Write-Host '  First question takes ~90s (model load); after that 45-60s.'
    Write-Host "  Stop with:  .\Start-JmpDocs.ps1 -Stop"
} else {
    Write-Warning 'the app did not come up. Last log lines:'
    Get-Content $Log, "$Log.err" -Tail 20 -ErrorAction SilentlyContinue
    exit 1
}
