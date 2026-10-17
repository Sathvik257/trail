@echo off
chcp 65001 >nul
".\.venv\Scripts\python.exe" -X utf8 "src\self_check.py"
