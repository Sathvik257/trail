param(
    [string]$PythonPath = 'C:\Users\sweth\Downloads\UROP\.venv\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$trainingProcessId = [int](Get-Content -LiteralPath 'outputs\telugu_llm_train.pid' -Raw)
while (Get-Process -Id $trainingProcessId -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}
if (-not (Select-String -LiteralPath 'outputs\telugu_llm_train.log' -Pattern 'Model saved:' -Quiet)) {
    throw 'Training exited without completing. See outputs\telugu_llm_train.log and .err.log.'
}
& $PythonPath -X utf8 -u src\evaluate_telugu_llm_checkpoint.py
if ($LASTEXITCODE -ne 0) { throw "Test evaluation exited with code $LASTEXITCODE" }
