param(
    [string]$Text = "ఆరోగ్యంగా ఉండటానికి",
    [ValidateSet('lstm', 'bilstm')][string]$Model = 'lstm',
    [int]$TopK = 5,
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
& $PythonPath -X utf8 src\telugu_neural_predictor.py --model $Model --text $Text --top-k $TopK
if ($LASTEXITCODE -ne 0) { throw "Prediction failed with exit code $LASTEXITCODE" }
