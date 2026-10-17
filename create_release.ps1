param(
    [string]$OutputZip = "Indic_FL_Word_Sentence_Final.zip"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutputPath = Join-Path $ProjectRoot $OutputZip

$includeRoots = @(
    "data",
    "data_hindi",
    "data_english",
    "outputs",
    "src"
)

$includeFiles = @(
    ".gitignore",
    "README.md",
    "PROJECT_REPORT.md",
    "FINAL_DELIVERY.md",
    "requirements.txt",
    "run_app.bat",
    "run_app.ps1",
    "run_self_check.bat",
    "run_self_check.ps1",
    "run_telugu_predict.bat",
    "run_telugu_predict.ps1",
    "run_hindi_predict.bat",
    "run_hindi_predict.ps1",
    "run_english_predict.bat",
    "run_english_predict.ps1",
    "create_release.ps1"
)

if (Test-Path -LiteralPath $OutputPath) {
    Remove-Item -LiteralPath $OutputPath -Force
}

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::Open($OutputPath, [System.IO.Compression.ZipArchiveMode]::Create)

try {
    foreach ($rootName in $includeRoots) {
        $rootPath = Join-Path $ProjectRoot $rootName
        if (-not (Test-Path -LiteralPath $rootPath)) {
            continue
        }

        Get-ChildItem -LiteralPath $rootPath -Recurse -File | ForEach-Object {
            if ($_.FullName -like "*\__pycache__\*" -or $_.Extension -in @(".pyc", ".pyo")) {
                return
            }

            $relative = $_.FullName.Substring($ProjectRoot.Length)
            while ($relative.StartsWith("\") -or $relative.StartsWith("/")) {
                $relative = $relative.Substring(1)
            }
            $relative = $relative.Replace("\", "/")
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $zip,
                $_.FullName,
                $relative,
                [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    }

    foreach ($fileName in $includeFiles) {
        $filePath = Join-Path $ProjectRoot $fileName
        if (Test-Path -LiteralPath $filePath) {
            $relative = $fileName.Replace("\", "/")
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $zip,
                $filePath,
                $relative,
                [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    }
}
finally {
    $zip.Dispose()
}

Write-Host "Created release package: $OutputPath"
