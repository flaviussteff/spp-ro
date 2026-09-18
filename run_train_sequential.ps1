# ==============================================================================
# SPP-Ro: Sequential Pretraining Orchestrator (Step 4 -> Step 5)
# Executes Step 4 (Baseline 125M) and upon completion spawns Step 5 (SPP 125M)
# ==============================================================================

$Host.UI.RawUI.WindowTitle = "SPP-Ro Pretraining - Step 4: Baseline Model"
Set-Location "C:\Users\Flavius Stefan\Desktop\licenta"

Write-Host "`n============================================================================" -ForegroundColor Cyan
Write-Host " [STEP 4/6] STARTING BASELINE MODEL PRETRAINING (10,000 STEPS)" -ForegroundColor Cyan
Write-Host " Device: NVIDIA GeForce RTX 3060 (12GB GDDR6, BF16 Active)" -ForegroundColor Cyan
Write-Host " Updates: Every 10 minutes with live token generation" -ForegroundColor Cyan
Write-Host " Checkpoints: models\base_ro_125m" -ForegroundColor Cyan
Write-Host "============================================================================`n" -ForegroundColor Cyan

# Run Step 4 (Baseline)
py src/train_pretrain.py --mode base --max-steps 10000 --update-interval-mins 10

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n`n============================================================================" -ForegroundColor Green
    Write-Host " [STEP 4 COMPLETED SUCCESSFULLY!]" -ForegroundColor Green
    Write-Host " Checkpoints saved to models\base_ro_125m" -ForegroundColor Green
    Write-Host " Now opening a separate terminal window for Step 5: Constitutional SPP..." -ForegroundColor Green
    Write-Host "============================================================================`n" -ForegroundColor Green

    # Launch Step 5 in a separate visible terminal window
    Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", "
        `$Host.UI.RawUI.WindowTitle = 'SPP-Ro Pretraining - Step 5: Constitutional SPP';
        Set-Location 'C:\Users\Flavius Stefan\Desktop\licenta';
        Write-Host '`n============================================================================' -ForegroundColor Magenta;
        Write-Host ' [STEP 5/6] STARTING CONSTITUTIONAL SPP PRETRAINING (10,000 STEPS)' -ForegroundColor Magenta;
        Write-Host ' Device: NVIDIA GeForce RTX 3060 (12GB GDDR6, BF16 Active)' -ForegroundColor Magenta;
        Write-Host ' Causal Attention Blocking & RoPE Aliasing Active' -ForegroundColor Magenta;
        Write-Host ' Updates: Every 10 minutes with live token generation' -ForegroundColor Magenta;
        Write-Host ' Checkpoints: models\spp_ro_125m' -ForegroundColor Magenta;
        Write-Host '============================================================================`n' -ForegroundColor Magenta;
        py src/train_pretrain.py --mode spp --max-steps 10000 --update-interval-mins 10
    "
} else {
    Write-Host "`n[ERROR] Step 4 exited with error code $LASTEXITCODE. Step 5 will not launch automatically." -ForegroundColor Red
}
