@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" v3turbo_desktop.py
) else (
    uv run --extra srt-quality --extra video python v3turbo_desktop.py
)
if errorlevel 1 pause
