param(
    [string]$Text = "I am learning English",
    [int]$TopK = 5,
    [switch]$Sentences
)

[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$argsList = @("-X", "utf8", "src\predict.py", "--language", "english", "--text", $Text, "--top-k", $TopK)
if ($Sentences) {
    $argsList += "--sentences"
}

& ".\.venv\Scripts\python.exe" @argsList
