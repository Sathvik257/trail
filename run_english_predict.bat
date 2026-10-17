@echo off
chcp 65001 >nul
".\.venv\Scripts\python.exe" -X utf8 "src\predict.py" --language english --text "%~1" --top-k 5 --sentences
