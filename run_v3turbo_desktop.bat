@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "from importlib.metadata import version; from importlib.util import find_spec; assert tuple(int(x) for x in version('moviepy').split('.')[:2]) >= (2, 2); assert all(find_spec(x) is not None for x in ('PIL', 'imageio_ffmpeg', 'proglog'))" >nul 2>&1
    if errorlevel 1 (
        echo Installing missing video dependencies...
        uv pip install --python ".venv\Scripts\python.exe" "moviepy>=2.2,<3"
        if errorlevel 1 echo Video dependency installation failed. Other tools can still start.
    )
    ".venv\Scripts\python.exe" v3turbo_desktop.py %*
) else (
    uv run --extra srt-quality --extra video python v3turbo_desktop.py %*
)
if errorlevel 1 pause
