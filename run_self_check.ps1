param([string]$PythonPath, [switch]$RequireHealth)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not $PythonPath) { $PythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Create .venv and install requirements.txt, or supply -PythonPath.' }
& $PythonPath -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
& $PythonPath -X utf8 tests/run_checks.py
if ($LASTEXITCODE -ne 0) { throw 'Project checks failed.' }
if ($RequireHealth) {
    $response = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8501/_stcore/health' -TimeoutSec 10
    if ($response.Content.Trim() -ne 'ok') { throw 'Streamlit health check failed.' }
}
