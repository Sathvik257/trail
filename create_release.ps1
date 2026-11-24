param([string]$OutputZip = 'next-word-prediction.zip')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$outputPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $OutputZip))
if (Test-Path -LiteralPath $outputPath) { throw "Output already exists: $outputPath" }
& git archive --format=zip "--output=$outputPath" HEAD
if ($LASTEXITCODE -ne 0) { throw 'Git archive failed.' }
Write-Output "Exported committed HEAD to $outputPath. Uncommitted changes are not included."
