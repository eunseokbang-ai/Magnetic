@echo off
REM run.bat launches this in a separate minimized window instead of a fixed
REM delay, so the browser only opens once the server actually responds.
REM Uses curl.exe (built into Windows since the 1803 update) rather than a
REM hidden PowerShell process - some antivirus/EDR software flags
REM "powershell -WindowStyle Hidden -ExecutionPolicy Bypass" as suspicious
REM and silently kills it, which would make the browser never open with no
REM visible error. --noproxy "*" also makes sure a system-wide HTTP proxy
REM (common on corporate machines) can't intercept this localhost check and
REM make every attempt fail even though the server is actually up.
setlocal
set COUNT=0

:loop
curl -s -f --noproxy "*" -o nul http://127.0.0.1:8000/api/health
if not errorlevel 1 (
    start "" "http://127.0.0.1:8000/"
    exit /b 0
)

set /a COUNT+=1
if %COUNT% GEQ 90 (
    echo [Magnetic] 서버가 90초 안에 응답하지 않았습니다. run.bat 창의 오류 메시지를 확인하세요.
    pause
    exit /b 1
)

timeout /t 1 /nobreak >nul
goto loop
