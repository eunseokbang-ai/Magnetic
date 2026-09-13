@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================
echo   자력자료 파일 - 취득 날짜별 목록 만들기
echo ============================================
echo.

REM 대상 폴더: 이 창에 드래그해 넣거나 인자로 주면 그것을, 없으면 아래 기본값을 사용.
set "TARGET=%~1"
if "!TARGET!"=="" set "TARGET=C:\magnetic\HaeNam_Mag\Magnetometer"

if not exist "!TARGET!" (
    echo [오류] 폴더를 찾을 수 없습니다: !TARGET!
    echo.
    echo        이 배치파일에 자료 폴더를 드래그해서 놓거나,
    echo        명령창에서 경로를 인자로 지정해 실행하세요:
    echo          자료목록_만들기.bat "D:\다른경로\Magnetometer"
    echo.
    pause
    exit /b 1
)

REM run.bat이 만들어 둔 가상환경을 우선 사용 (pandas 등이 이미 설치돼 있음).
set "PYEXE=backend\venv\Scripts\python.exe"
if not exist "!PYEXE!" (
    echo [알림] backend\venv 가 없어 시스템 python 을 사용합니다.
    echo        pandas 관련 오류가 나면 run.bat 을 한 번 실행해 설치를 끝낸 뒤 다시 시도하세요.
    echo.
    set "PYEXE=python"
)

echo 대상 폴더: !TARGET!
echo.
echo [참고] 장비가 UTC로 기록하는 경우 한국시간 기준으로 정리하려면
echo        이 창을 닫고 아래처럼 실행하세요 (9시간 이동):
echo          !PYEXE! tools\inventory_mag_files.py "!TARGET!" --tz-shift-hours 9
echo.

"!PYEXE!" tools\inventory_mag_files.py "!TARGET!"
if errorlevel 1 (
    echo.
    echo [오류] 목록 생성에 실패했습니다. 위 메시지를 확인하세요.
)

echo.
pause
