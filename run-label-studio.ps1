# Launch Label Studio with local file serving enabled for the project directory
$ProjectRoot = $PSScriptRoot
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = $ProjectRoot

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host " Starting Label Studio in .venv" -ForegroundColor Green
Write-Host " Document Root: $ProjectRoot" -ForegroundColor Yellow
Write-Host " Local File Serving: Enabled" -ForegroundColor Yellow
Write-Host " Default URL: http://localhost:8080" -ForegroundColor Cyan
Write-Host " Press Ctrl+C to stop the server." -ForegroundColor Gray
Write-Host "===================================================" -ForegroundColor Cyan

& "$ProjectRoot\.venv\Scripts\label-studio.exe" @args
