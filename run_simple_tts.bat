@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" simple_tts.py %*
) else (
    uv run python simple_tts.py %*
)
set "SIMPLE_TTS_EXIT_CODE=%ERRORLEVEL%"
if not "%SIMPLE_TTS_EXIT_CODE%"=="0" pause
exit /b %SIMPLE_TTS_EXIT_CODE%
