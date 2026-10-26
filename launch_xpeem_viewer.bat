@echo off
call conda activate xpeem-viewer
python "%~dp0xpeem_viewer.py"

if errorlevel 1 (
    echo.
    echo XPEEM Viewer encountered an error.
    echo Press any key to close this window.
    pause >nul
)
