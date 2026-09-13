@echo off
rem 설치파일/매뉴얼까지 마무리된 기존 버전을 실행한다.
rem 브랜치 전환, git pull, 필요할 때의 재빌드까지 _launch.bat이 알아서 한다.
call "%~dp0_launch.bat" claude/magnetic-survey-app-dev-08870i "기존 버전"
