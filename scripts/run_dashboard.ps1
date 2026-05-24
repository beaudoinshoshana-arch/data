$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Frontend = Join-Path $Root "dashboard\frontend"

Write-Host "Starting WWTP Safe-MARL dashboard..."
Write-Host "Backend:  http://127.0.0.1:8000"
Write-Host "Frontend: http://127.0.0.1:5173"

Start-Process -FilePath "python" -ArgumentList "-m","uvicorn","dashboard.backend.main:app","--host","127.0.0.1","--port","8000" -WorkingDirectory $Root -WindowStyle Hidden
Start-Sleep -Seconds 3
Start-Process -FilePath "npm.cmd" -ArgumentList "run","dev","--","--host","127.0.0.1","--port","5173" -WorkingDirectory $Frontend -WindowStyle Hidden

Write-Host "Services launched. Open http://127.0.0.1:5173"
