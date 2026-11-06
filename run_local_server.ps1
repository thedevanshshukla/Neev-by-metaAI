# ==============================================================================
# NEEV: 24/7 Dedicated Local Server Launcher with RAM Guardrail & Never-Sleep
# ==============================================================================

Write-Host "=== [1/3] Enforcing Windows Never-Sleep & Lid-Close Settings ===" -ForegroundColor Cyan
powercfg -change -standby-timeout-ac 0
powercfg -change -standby-timeout-dc 0
powercfg -change -hibernate-timeout-ac 0
powercfg -change -hibernate-timeout-dc 0
# Configure Lid Close Action -> 0 (Do Nothing) on both AC and Battery
powercfg /setacvalueindex SCHEME_CURRENT 4f971e89-eebd-4455-a8de-9e59040e7347 5ca83367-6e45-459f-a27b-476b1d01c936 0 2>$null
powercfg /setdcvalueindex SCHEME_CURRENT 4f971e89-eebd-4455-a8de-9e59040e7347 5ca83367-6e45-459f-a27b-476b1d01c936 0 2>$null
powercfg /setactive SCHEME_CURRENT
Write-Host "System sleep and lid-close power settings applied: machine will stay online with lid closed." -ForegroundColor Green

Write-Host "`n=== [2/3] Setting 4 GB Peak RAM Safety Guardrail ===" -ForegroundColor Cyan
$env:NEEV_MAX_RAM_MB = "4096"
$env:HOST = "0.0.0.0"
$env:PORT = "8000"
Write-Host "NEEV_MAX_RAM_MB set to 4096 MB (4 GB). Safe OOM fallback active." -ForegroundColor Green

Write-Host "`n=== [3/3] Starting Neev Production Server & Cloudflare Tunnel ===" -ForegroundColor Cyan
# Start Uvicorn in background
$uvicornProcess = Start-Process -FilePath "python" -ArgumentList "-m uvicorn neev.app.main:app --host 0.0.0.0 --port 8000" -PassThru -NoNewWindow
Write-Host "Neev API server running on http://localhost:8000 (PID: $($uvicornProcess.Id))" -ForegroundColor Green

# Start Cloudflared tunnel
if (Test-Path ".\cloudflared.exe") {
    Write-Host "Starting Cloudflare 24/7 public tunnel..." -ForegroundColor Cyan
    $cfLog = "$PSScriptRoot\cloudflared.log"
    Remove-Item $cfLog -ErrorAction SilentlyContinue
    $cfProcess = Start-Process -FilePath ".\cloudflared.exe" -ArgumentList "tunnel --url http://localhost:8000" -RedirectStandardError $cfLog -PassThru -NoNewWindow
    
    $publicUrl = "Waiting for tunnel connection..."
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 1
        if (Test-Path $cfLog) {
            $match = Select-String -Path $cfLog -Pattern "https://[a-zA-Z0-9-]+\.trycloudflare\.com"
            if ($match) {
                $publicUrl = $match.Matches[0].Value
                break
            }
        }
    }
} else {
    Write-Host "cloudflared.exe not found in directory. Running locally on port 8000 only." -ForegroundColor Yellow
    $publicUrl = "Not available (cloudflared.exe missing)"
}

Write-Host "`n==========================================================================" -ForegroundColor Yellow
Write-Host "  NEEV IS ACTIVE 24/7 ON YOUR LAPTOP WITH 4 GB RAM CEILING" -ForegroundColor Yellow
Write-Host "  - Local URL:   http://localhost:8000" -ForegroundColor White
Write-Host "  - Public URL:  $publicUrl" -ForegroundColor Green
Write-Host "  - Never-Sleep: Laptop lid can be closed without interrupting server" -ForegroundColor White
Write-Host "  - RAM Guard:   If memory nears 4GB, server stays online with alert banner" -ForegroundColor White
Write-Host "==========================================================================" -ForegroundColor Yellow
Write-Host "`nServer is running. Minimize this window to keep it running in background." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server.`n" -ForegroundColor Gray

# Keep script running to maintain child processes
try {
    Wait-Process -Id $uvicornProcess.Id
} finally {
    if ($cfProcess -and -not $cfProcess.HasExited) {
        Stop-Process -Id $cfProcess.Id -Force -ErrorAction SilentlyContinue
    }
}

