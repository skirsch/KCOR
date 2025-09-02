# PowerShell script to run KCOR analysis

Write-Host "Running KCOR analysis..." -ForegroundColor Green
Write-Host "=" * 50

try {
    python KCORv4.py ../../Czech/data/KCOR_output.xlsx KCOR_processed_REAL.xlsx
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nKCOR analysis completed successfully!" -ForegroundColor Green
        Write-Host "Output file: KCOR_processed_REAL.xlsx" -ForegroundColor Cyan
    } else {
        Write-Host "`nKCOR analysis failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host "`nError running KCOR analysis: $_" -ForegroundColor Red
}

Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
