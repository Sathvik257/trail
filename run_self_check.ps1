param(
    [switch]$RequireHealth,
    [string]$PythonPath
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not $PythonPath) {
    $PythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        $PythonPath = Join-Path $env:USERPROFILE 'Downloads\UROP\.venv\Scripts\python.exe'
    }
}
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Pass -PythonPath to an environment with requirements.txt installed.' }
& $PythonPath -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
& $PythonPath -m compileall -q src
if ($LASTEXITCODE -ne 0) { throw 'Python syntax check failed.' }
& $PythonPath -X utf8 src\check_telugu_comparison.py
if ($LASTEXITCODE -ne 0) { throw 'Model regression checks failed.' }
& $PythonPath -X utf8 src\check_telugu_ui.py
if ($LASTEXITCODE -ne 0) { throw 'Checkpoint/interface checks failed.' }
if ($RequireHealth) {
    $response = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8501/_stcore/health' -TimeoutSec 10
    if ($response.Content.Trim() -ne 'ok') { throw 'Web preview health check failed.' }
}
Write-Output 'All current Telugu-phase checks passed. Training completion is reported separately in outputs\telugu_comparison\status.json.'
