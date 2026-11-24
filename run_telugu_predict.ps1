param(
    [Parameter(Mandatory=$true)][string]$Text,
    [ValidateSet('lstm', 'bilstm')][string]$Model = 'lstm',
    [int]$TopK = 5,
    [string]$PythonPath
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not $PythonPath) { $PythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Create .venv and install requirements.txt, or supply -PythonPath.' }
& $PythonPath -X utf8 src/predict.py --language telugu --model $Model --text $Text --top-k $TopK
if ($LASTEXITCODE -ne 0) { throw "Prediction failed with exit code $LASTEXITCODE" }
