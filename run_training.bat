@echo off
title SPP-Ro Pretraining Pipeline (RTX 3060 12GB)
cd /d "%~dp0"

echo ============================================================================
echo  SPP-Ro: Sequential Pretraining Pipeline (Token Zero)
echo  Hardware: NVIDIA GeForce RTX 3060 12GB GDDR6 (BF16 Active)
echo  Updates: Every 10 minutes with live token generation
echo ============================================================================
echo.
echo [1/2] STARTING STEP 4: BASELINE MODEL PRETRAINING (10,000 STEPS)...
echo Checkpoints will be saved to: models\base_ro_125m
echo.

py src/train_pretrain.py --mode base --max-steps 10000 --update-interval-mins 10

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Step 4 exited with error code %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================================
echo  [STEP 4 COMPLETE!] Baseline model saved to models\base_ro_125m
echo  NOW LAUNCHING STEP 5: CONSTITUTIONAL SPP MODEL IN A SEPARATE TERMINAL...
echo ============================================================================
echo.

start "SPP-Ro Step 5: Constitutional SPP Model" cmd /k "cd /d "%~dp0" && echo [2/2] STARTING STEP 5: CONSTITUTIONAL SPP PRETRAINING (10,000 STEPS)... && py src/train_pretrain.py --mode spp --max-steps 10000 --update-interval-mins 10"

echo Step 4 is finished, and Step 5 has been launched in its own dedicated window!
pause
