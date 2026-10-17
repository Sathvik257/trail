param(
    [string]$PythonPath = (Join-Path $PSScriptRoot '.venv\Scripts\python.exe'),
    [int]$Epochs = 3,
    [switch]$Background
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw 'Pass -PythonPath with the path to a Python environment containing requirements.txt dependencies.'
}
if (-not (Test-Path -LiteralPath 'data_telugu_llm\provenance.json')) {
    throw 'Prepare the downloaded corpora with src\prepare_telugu_llm_data.py first.'
}
if (Test-Path -LiteralPath 'outputs\telugu_llm_lstm.pt') {
    throw 'A training checkpoint already exists. Keep it and use centralized_train.py with an explicit output/resume path.'
}

$trainingArgs = @(
    '-X', 'utf8', '-u', 'src\centralized_train.py',
    '--language', 'telugu', '--data-dir', 'data_telugu_llm',
    '--validation-data-dir', 'data_telugu_llm\validation',
    '--provenance', 'data_telugu_llm\provenance.json',
    '--epochs', "$Epochs", '--batch-size', '128',
    '--embedding-dim', '64', '--hidden-dim', '128',
    '--num-layers', '1', '--dropout', '0', '--threads', '2',
    '--shard-cache-size', '1', '--log-every', '50',
    '--save-every-batches', '100', '--output', 'outputs\telugu_llm_lstm.pt'
)
if ($Background) {
    $process = Start-Process -WindowStyle Hidden -FilePath $PythonPath -ArgumentList $trainingArgs `
        -WorkingDirectory $PSScriptRoot -RedirectStandardOutput 'outputs\telugu_llm_train.log' `
        -RedirectStandardError 'outputs\telugu_llm_train.err.log' -PassThru
    $process.Id | Set-Content -LiteralPath 'outputs\telugu_llm_train.pid'
    Write-Output "Training started: PID $($process.Id). Log: outputs\telugu_llm_train.log"
} else {
    & $PythonPath @trainingArgs
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
}
