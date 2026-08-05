@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================
echo   Magnetic - 드론 자력탐사 자료 처리 프로그램
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않거나 PATH에 등록되어 있지 않습니다.
    echo        https://www.python.org 에서 Python 3.11 이상을 설치한 뒤 다시 실행하세요.
    echo        설치 화면에서 "Add python.exe to PATH" 옵션을 반드시 체크하세요.
    echo.
    pause
    exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
    echo [오류] Node.js가 설치되어 있지 않거나 PATH에 등록되어 있지 않습니다.
    echo        https://nodejs.org 에서 LTS 버전을 설치한 뒤 다시 실행하세요.
    echo.
    pause
    exit /b 1
)

echo [1/4] 최신 버전 확인 중 (git pull)...
git rev-parse --is-inside-work-tree >nul 2>nul
if errorlevel 1 (
    echo   - git 저장소가 아닙니다. 업데이트 확인을 건너뜁니다.
) else (
    git pull
    if errorlevel 1 (
        echo   [경고] 최신 버전을 받아오지 못했습니다. 인터넷 연결 또는 로컬에서 직접 수정한
        echo          파일이 있는지 확인하세요. 일단 지금 있는 버전으로 계속 진행합니다.
    )
)
echo.

set MARKER_FILE=.last_build_commit
set CURRENT_COMMIT=unknown
for /f "delims=" %%i in ('git rev-parse HEAD 2^>nul') do set CURRENT_COMMIT=%%i

set NEED_BUILD=0
if not exist "%MARKER_FILE%" set NEED_BUILD=1
if not exist "backend\venv\Scripts\python.exe" set NEED_BUILD=1
if not exist "frontend\dist\index.html" set NEED_BUILD=1
if not exist "frontend\node_modules" set NEED_BUILD=1

if exist "%MARKER_FILE%" (
    set /p LAST_COMMIT=<"%MARKER_FILE%"
    if not "!LAST_COMMIT!"=="!CURRENT_COMMIT!" set NEED_BUILD=1
)

if "!NEED_BUILD!"=="1" (
    echo [2/4] 처음 실행이거나 새 버전이 있어 설치/빌드를 진행합니다 - 수 분 걸릴 수 있습니다...
    echo.

    if not exist "backend\venv\Scripts\python.exe" (
        echo   - Python 가상환경 생성 중...
        python -m venv backend\venv
        if errorlevel 1 (
            echo [오류] 가상환경 생성에 실패했습니다.
            pause
            exit /b 1
        )
    )

    echo   - 백엔드 패키지 설치 중...
    backend\venv\Scripts\python.exe -m pip install --quiet --upgrade pip
    backend\venv\Scripts\python.exe -m pip install --quiet -r backend\requirements.txt
    if errorlevel 1 (
        echo [오류] 백엔드 패키지 설치에 실패했습니다. 위 오류 메시지를 확인하세요.
        pause
        exit /b 1
    )

    echo   - 프론트엔드 패키지 설치 중...
    pushd frontend
    call npm install --silent
    if errorlevel 1 (
        echo [오류] 프론트엔드 패키지 설치에 실패했습니다.
        popd
        pause
        exit /b 1
    )

    echo   - 프론트엔드 빌드 중...
    call npm run build
    if errorlevel 1 (
        echo [오류] 프론트엔드 빌드에 실패했습니다.
        popd
        pause
        exit /b 1
    )
    popd

    echo !CURRENT_COMMIT!> "%MARKER_FILE%"
    echo.
    echo   설치/빌드 완료.
) else (
    echo [2/4] 이미 최신 상태입니다 - 설치/빌드를 건너뜁니다.
)
echo.

echo [3/4] 서버를 시작합니다. 잠시 후 브라우저가 자동으로 열립니다.
echo        이 창을 닫으면 프로그램이 종료됩니다.
echo.

start "" cmd /c "timeout /t 6 /nobreak >nul & start http://127.0.0.1:8000/"

echo [4/4] 실행 중... (종료하려면 이 창을 닫거나 Ctrl+C)
echo ============================================
backend\venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000

echo.
echo 프로그램이 종료되었습니다.
pause
