param(
    [int]$Port = 8501,
    [string]$PythonPath
)

[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$ErrorActionPreference = 'Stop'
if (-not $PythonPath) {
    $PythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        # Reuse the existing environment when this project was copied from Downloads.
        $PythonPath = Join-Path $env:USERPROFILE 'Downloads\UROP\.venv\Scripts\python.exe'
    }
}
if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw 'Python environment missing. Create .venv and install requirements.txt, or pass -PythonPath with the full path to a Python executable that has Streamlit installed.'
}
Write-Host "Using Python: $PythonPath"
& $PythonPath -m streamlit run (Join-Path $PSScriptRoot 'src\telugu_app.py') --server.port $Port
exit $LASTEXITCODE

