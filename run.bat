@echo off
rem 개발 중인 새 버전(자료처리 기능 개선)을 실행한다.
rem 브랜치 전환, git pull, 필요할 때의 재빌드까지 _launch.bat이 알아서 한다.
call "%~dp0_launch.bat" claude/magnetic-processing-improve "새 버전 (자료처리 기능 개선)"
