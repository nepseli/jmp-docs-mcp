<#
.SYNOPSIS
    Starts the JMP Docs Assistant chat app, and makes sure Ollama is up first.

.DESCRIPTION
    Run this instead of typing the streamlit command by hand. It:
      - starts the Ollama server if it is not already listening
      - starts Streamlit detached, so it keeps running after you close the window
      - waits until the app actually answers, rather than assuming it worked
      - refuses to start a second copy if one is already running

.EXAMPLE
    .\Start-JmpDocs.ps1
    .\Start-JmpDocs.ps1 -Stop
    .\Start-JmpDocs.ps1 -Status
#>
[CmdletBinding()]
param(
    [int]$Port = 8501,
    [switch]$Stop,
    [switch]$Status
)

$ErrorActionPreference = 'Stop'

$Proj      = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python    = Join-Path $Proj '.venv\Scripts\python.exe'
$OllamaExe = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
$Log       = Join-Path $env:TEMP 'jmpdocs-streamlit.log'
$PidFile   = Join-Path $env:TEMP 'jmpdocs-streamlit.pid'

function Test-Port([int]$p) {
    [bool](Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue)
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
    "app    ($Port) : $(if (Test-Port $Port)  { 'running' } else { 'stopped' })"
    $appPid = Get-AppPid
    if ($appPid) { "app pid        : $appPid" }
    "url            : http://localhost:$Port"
    "log            : $Log"
    return
}

# ------------------------------------------------------------------ stop ----
if ($Stop) {
    $appPid = Get-AppPid
    if ($appPid) {
        Stop-Process -Id $appPid -Force -ErrorAction SilentlyContinue
        Remove-Item $PidFile -ErrorAction SilentlyContinue
        "stopped app (pid $appPid)"
    }
    # catch any stray listener on the port too
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conn) {
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
        "stopped stray listener on port $Port (pid $($c.OwningProcess))"
    }
    if (-not $appPid -and -not $conn) { "nothing was running on port $Port" }
    return
}

# ----------------------------------------------------------------- start ----
if (-not (Test-Path $Python)) {
    throw "virtualenv not found at $Python. Run:  python -m venv .venv ;  .\.venv\Scripts\python.exe -m pip install -e `".[dev]`""
}

if (Test-Port $Port) {
    "app already running -> http://localhost:$Port"
    return
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
