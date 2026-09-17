@echo off
REM Double-click this to launch the JMP Docs Assistant.
REM It starts Ollama if needed, starts the app, and opens your browser.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-JmpDocs.ps1"
if errorlevel 1 (
    echo.
    echo The app failed to start. See the messages above.
    pause
    exit /b 1
)
start "" http://localhost:8501
