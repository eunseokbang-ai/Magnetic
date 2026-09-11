@echo off
setlocal
cd /d "%~dp0"

REM 이 파일은 CP949(한국어 Windows 콘솔 기본 인코딩)로 저장되어 있습니다.
REM UTF-8 로 저장하고 chcp 65001 을 쓰면, cmd 가 코드페이지를 바꾼 뒤
REM 파일을 잘못된 바이트 위치에서 이어 읽어 명령을 한 줄 중간에서 잘라
REM 버립니다. 명령이 반토막 난 채 실행되어, 잘린 조각마다 내부 또는
REM 외부 명령이 아니라는 오류가 줄줄이 뜹니다.
REM 편집할 때는 반드시 ANSI/CP949 로 저장하세요. UTF-8 로 저장하면 깨집니다.

echo ================================================
echo   MagComp 열순서 - MagLPF 바로 뒤로 옮기기
echo   Mag, MagLPF, MagComp 순서가 되도록 바꿉니다
echo ================================================
echo.

REM 대상 폴더: 이 배치파일에 폴더를 드래그해 놓거나 인자로 주면 그것을,
REM 없으면 아래 기본값을 씁니다.
set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=D:\HaeNam_Mag\Magnetometer_old2_comp_mod"
if not exist "%TARGET%" goto nofolder

REM 표준 라이브러리만 쓰므로 venv 없이도 돌아가지만, 있으면 그걸 씁니다.
set "PYEXE=backend\venv\Scripts\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

REM 저장소 안이라면 tools\ 에, 이 bat 만 따로 복사해 쓴다면 같은 폴더에.
set "SCRIPT=tools\reorder_magcomp_column.py"
if not exist "%SCRIPT%" set "SCRIPT=reorder_magcomp_column.py"
if not exist "%SCRIPT%" goto noscript

echo 대상 폴더: %TARGET%
echo.
echo 먼저 무엇이 바뀔지만 확인합니다. 파일을 쓰지 않습니다.
echo ----------------------------------------------------------------
"%PYEXE%" "%SCRIPT%" "%TARGET%" --dry-run
if errorlevel 1 goto failpreview
echo ----------------------------------------------------------------
echo.
echo 위 내용대로 열 순서를 바꿔 저장할까요?
echo 원본 파일은 그대로 두고 "원래이름-r.csv" 로 새로 만듭니다.
echo.
set "GO="
set /p "GO=진행하려면 Y 를 누르고 Enter, 취소는 그냥 Enter: "
if /i not "%GO%"=="Y" goto cancelled

echo.
"%PYEXE%" "%SCRIPT%" "%TARGET%"
if errorlevel 1 goto failwrite

echo.
echo [참고] 다른 열 이름이라면 아래처럼 지정해 실행하세요.
echo        %PYEXE% "%SCRIPT%" "%TARGET%" --move MagComp --after MagLPF
echo.
pause
exit /b 0

:nofolder
echo [오류] 폴더를 찾을 수 없습니다: %TARGET%
echo.
echo        이 배치파일에 자료 폴더를 드래그해서 놓거나,
echo        명령창에서 경로를 인자로 지정해 실행하세요:
echo          MagComp_열순서.bat "D:\다른경로\Magnetometer"
echo.
pause
exit /b 1

:noscript
echo [오류] reorder_magcomp_column.py 를 찾을 수 없습니다.
echo        이 bat 과 같은 폴더, 또는 tools\ 폴더 안에 두세요.
echo.
pause
exit /b 1

:failpreview
echo.
echo [오류] 확인 단계에서 실패했습니다. 위 메시지를 보세요.
echo.
pause
exit /b 1

:cancelled
echo.
echo 취소했습니다. 파일을 하나도 바꾸지 않았습니다.
echo.
pause
exit /b 0

:failwrite
echo.
echo [오류] 저장에 실패했습니다. 위 메시지를 확인하세요.
echo.
pause
exit /b 1
