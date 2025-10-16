Param(
    [string]$PromptJson = "templates/son/prompt.json",
    [string]$ContextFile = "templates/son/fukuma_sun_latex.txt"
)

if (!(Test-Path $PromptJson)) {
    Write-Error "prompt.json not found: $PromptJson"
    exit 1
}
if (!(Test-Path $ContextFile)) {
    Write-Error "context file not found: $ContextFile"
    exit 1
}

$json = Get-Content $PromptJson -Raw | ConvertFrom-Json
$content = Get-Content $ContextFile -Raw
$nl = [Environment]::NewLine
$marker = "BEGIN CONTEXT ($ContextFile)"
$end = "END CONTEXT"

if ($json.task_description -notmatch [regex]::Escape($marker)) {
    Copy-Item $PromptJson "$PromptJson.bak" -Force
    $block = $nl + $nl + $marker + $nl + $content + $nl + $end + $nl
    $json.task_description = $json.task_description + $block
    $json | ConvertTo-Json -Depth 10 | Set-Content $PromptJson -Encoding UTF8
    Write-Host "Appended context to $PromptJson. Backup at $PromptJson.bak"
} else {
    Write-Host "Context already present; skipped."
}

