@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ================================================
echo   MagComp 열순서 - MagLPF 바로 뒤로 옮기기
echo   (Mag, MagLPF, MagComp 순서가 되도록)
echo ================================================
echo.

REM 대상 폴더: 이 창에 드래그해 넣거나 인자로 주면 그것을, 없으면 아래 기본값.
set "TARGET=%~1"
if "!TARGET!"=="" set "TARGET=D:\HaeNam_Mag\Magnetometer_old2_comp_mod"

if not exist "!TARGET!" (
    echo [오류] 폴더를 찾을 수 없습니다: !TARGET!
    echo.
    echo        이 배치파일에 자료 폴더를 드래그해서 놓거나,
    echo        명령창에서 경로를 인자로 지정해 실행하세요:
    echo          MagComp_열순서.bat "D:\다른경로\Magnetometer"
    echo.
    pause
    exit /b 1
)

REM 표준 라이브러리만 쓰므로 venv 없이도 동작하지만, 있으면 그걸 씁니다.
set "PYEXE=backend\venv\Scripts\python.exe"
if not exist "!PYEXE!" set "PYEXE=python"

REM 스크립트 위치: 저장소라면 tools\ 안에, 이 bat 만 따로 복사해 쓰는
REM 경우라면 bat 과 같은 폴더에 둡니다. 둘 다 지원합니다.
set "SCRIPT=tools\reorder_magcomp_column.py"
if not exist "!SCRIPT!" set "SCRIPT=reorder_magcomp_column.py"
if not exist "!SCRIPT!" (
    echo [오류] reorder_magcomp_column.py 를 찾을 수 없습니다.
    echo        이 bat 과 같은 폴더, 또는 tools\ 폴더 안에 두세요.
    echo.
    pause
    exit /b 1
)

echo 대상 폴더: !TARGET!
echo.
echo 먼저 무엇이 바뀔지만 확인합니다 (파일을 쓰지 않습니다).
echo ----------------------------------------------------------------
"!PYEXE!" "!SCRIPT!" "!TARGET!" --dry-run
if errorlevel 1 (
    echo.
    echo [오류] 확인 단계에서 실패했습니다. 위 메시지를 보세요.
    echo.
    pause
    exit /b 1
)

echo ----------------------------------------------------------------
echo.
echo 위 내용대로 열 순서를 바꿔 저장할까요?
echo   원본 파일은 그대로 두고 "원래이름-r.csv" 로 새로 만듭니다.
echo.
set "GO="
set /p "GO=진행하려면 Y 를 누르고 Enter (취소는 그냥 Enter): "
if /i not "!GO!"=="Y" (
    echo.
    echo 취소했습니다. 파일을 하나도 바꾸지 않았습니다.
    echo.
    pause
    exit /b 0
)

echo.
"!PYEXE!" "!SCRIPT!" "!TARGET!"
if errorlevel 1 (
    echo.
    echo [오류] 저장에 실패했습니다. 위 메시지를 확인하세요.
)

echo.
echo [참고] 다른 열 이름이라면 아래처럼 지정해 실행하세요:
echo          !PYEXE! "!SCRIPT!" "!TARGET!" --move MagComp --after MagLPF
echo.
pause
