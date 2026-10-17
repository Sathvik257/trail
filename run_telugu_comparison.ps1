param(
    [int]$Epochs = 1,
    [int]$Threads = 4,
    [ValidateSet('lstm', 'bilstm')][string]$FirstModel = 'lstm',
    [string]$PythonPath,
    [switch]$Background
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not $PythonPath) {
    $PythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        $PythonPath = Join-Path $env:USERPROFILE 'Downloads\UROP\.venv\Scripts\python.exe'
    }
}
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Pass -PythonPath pointing to Python with PyTorch installed.' }
$trainArgs = @('-X', 'utf8', '-u', 'src\train_telugu_comparison.py', '--epochs', "$Epochs", '--threads', "$Threads", '--first-model', $FirstModel)
if ($Background) {
    $out = Join-Path $PSScriptRoot 'outputs\telugu_comparison'
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $pidPath = Join-Path $out 'training.pid'
    if (Test-Path -LiteralPath $pidPath) {
        $existingProcessId = [int](Get-Content -LiteralPath $pidPath -Raw)
        if (Get-Process -Id $existingProcessId -ErrorAction SilentlyContinue) { throw "Training process $existingProcessId is still running." }
    }
    $process = Start-Process -WindowStyle Hidden -FilePath $PythonPath -ArgumentList $trainArgs -WorkingDirectory $PSScriptRoot `
        -RedirectStandardOutput (Join-Path $out 'training.log') -RedirectStandardError (Join-Path $out 'training.err.log') -PassThru
    $process.Id | Set-Content -LiteralPath $pidPath
    Write-Output "Training PID: $($process.Id). Progress: outputs\telugu_comparison\status.json"
} else {
    & $PythonPath @trainArgs
    if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
}

