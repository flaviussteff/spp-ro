@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Train SPP-Ro-125M (Token Zero - 50,000 Steps - Compute-Matched with Base-Ro)

echo ============================================================================
echo   SPP-Ro: Token Zero Pretraining - CONSTITUTIONAL MODEL (SPP-Ro-125M)
echo   Hardware: NVIDIA GeForce RTX 3060 12GB (BF16 / SDPA)
echo   Dataset: 60,000 constitutional reflections (50%% Sensitive + 50%% Factual)
echo   Interleaving Rate: 10%% SPP (Reflections) / 90%% Natural Text
echo   Configuration: 50,000 STEPS (3,200,000 texts, ~3.27B tokens) - COMPUTE MATCHED
echo   Estimated Time: ~48-50 hours (auto-checkpoint every 5,000 steps)
echo   After Training: Automatic upload to Hugging Face (flaviussteff/spp-ro-125m)
echo ============================================================================
echo.
echo The model will train from Token Zero (random initialization, seed=42)
echo under identical conditions with Base-Ro-125M for a rigorous scientific comparison.
echo.
echo Press any key to begin training...
pause >nul

echo.
echo [*] Starting SPP-Ro-125M pretraining (50,000 steps, hourly live diagnostics)...
py src/train_scale_pretrain.py --mode spp --steps 50000 --from-scratch --push-to-hub --interval-mins 60 --save-steps 5000
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] SPP model pretraining failed with exit code %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================================
echo   [SUCCESS] 50,000 steps pretraining and Hugging Face upload completed!
echo   Model available at: https://huggingface.co/flaviussteff/spp-ro-125m
echo ============================================================================
pause
