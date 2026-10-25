@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

rem MineMag(드론 자력탐사 지뢰·불발탄 탐지)을 실행한다.
rem DroneMag Studio의 _launch.bat과 같은 방식이되, 개발 중인 프로그램이라
rem 브랜치 전환과 git pull은 하지 않는다 - 지금 체크아웃된 코드를 그대로 쓴다.
rem Python 가상환경은 DroneMag Studio와 공유한다(backend\venv). 필요한 패키지가
rem 같고(magcore 공용), 두 번 설치할 이유가 없기 때문이다.

echo ============================================
echo   MineMag - 드론 자력탐사 지뢰/불발탄 탐지
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않거나 PATH에 등록되어 있지 않습니다.
    echo        https://www.python.org 에서 Python 3.11 이상을 설치한 뒤 다시 실행하세요.
    pause
    exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
    echo [오류] Node.js가 설치되어 있지 않거나 PATH에 등록되어 있지 않습니다.
    echo        https://nodejs.org 에서 LTS 버전을 설치한 뒤 다시 실행하세요.
    pause
    exit /b 1
)

echo [1/3] 설치/빌드 상태 확인 중...

set MARKER_FILE=.last_build_commit_minemag
set CURRENT_COMMIT=unknown
for /f "delims=" %%i in ('git rev-parse HEAD 2^>nul') do set CURRENT_COMMIT=%%i

set NEED_BUILD=0
if not exist "%MARKER_FILE%" set NEED_BUILD=1
if not exist "backend\venv\Scripts\python.exe" set NEED_BUILD=1
if not exist "minemag\frontend\dist\index.html" set NEED_BUILD=1
if not exist "minemag\frontend\node_modules" set NEED_BUILD=1

if exist "%MARKER_FILE%" (
    set /p LAST_COMMIT=<"%MARKER_FILE%"
    if not "!LAST_COMMIT!"=="!CURRENT_COMMIT!" set NEED_BUILD=1
)

if "!NEED_BUILD!"=="1" (
    echo [2/3] 설치/빌드를 진행합니다 - 처음에는 수 분 걸릴 수 있습니다...
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
    backend\venv\Scripts\python.exe -m pip install --quiet -r minemag\backend\requirements.txt
    if errorlevel 1 (
        echo [오류] 백엔드 패키지 설치에 실패했습니다. 위 오류 메시지를 확인하세요.
        pause
        exit /b 1
    )

    echo   - 프론트엔드 패키지 설치 중...
    pushd minemag\frontend
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
    echo [2/3] 이미 최신 상태입니다 - 설치/빌드를 건너뜁니다.
    echo        코드를 고친 뒤에는 커밋을 해야 다시 빌드합니다. 바로 반영하려면
    echo        %MARKER_FILE% 파일을 지우고 다시 실행하세요.
)
echo.

echo [3/3] 서버를 시작합니다. 준비되는 대로 브라우저가 자동으로 열립니다.
echo        포트 8100부터 비어 있는 포트를 자동으로 찾으므로 DroneMag Studio와
echo        같이 실행해도 됩니다. 이 창을 닫으면 프로그램이 종료됩니다.
echo ============================================
backend\venv\Scripts\python.exe minemag\backend\launcher.py

echo.
echo 프로그램이 종료되었습니다.
pause
