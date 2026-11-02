#!/bin/bash
# full build: pptx -> theme -> pdf -> qa pages
cd /c/magnetic/docs/seminar
export PPTX_SKILL_DIR="C:/Users/Eunseok Bang/AppData/Local/Claude-Data/claude_pzs8sxrjxfjjc/local-agent-mode-sessions/skills-plugin/886b004f-3df7-44b8-9fba-19d56b775716/d0c8f276-23d6-482a-9864-3ac9b7b5d9d9/skills/pptx"
export NODE_PATH="C:/magnetic/docs/seminar/node_modules"
node build_deck.js 2>&1 | grep -v "^\s*at " | tail -3
powershell -NoProfile -ExecutionPolicy Bypass -File export_pdf.ps1 -Pptx "C:\magnetic\docs\seminar\DroneMag_Studio_세미나.pptx" -Pdf "C:\magnetic\docs\seminar\DroneMag_Studio_세미나.pdf" 2>&1 | tail -1
python render_pages.py "DroneMag_Studio_세미나.pdf" 70
