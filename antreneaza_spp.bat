@echo off
cd /d "%~dp0"
title SPP-Ro-125M Scale Training

echo ============================================================================
echo   SPP-Ro: Antrenare Scalata Independenta - Model CONSTITUTIONAL (SPP-Ro-125M)
echo   Hardware: NVIDIA GeForce RTX 3060 12GB (BF16)
echo   Integrare reflexii constitutionale live (Attention Blocking + Sidecar)
echo ============================================================================
echo.
echo Model: SPP-Ro-125M (+50.000 pasi, max 60.000)
echo NOTA: Acest script antreneaza DOAR modelul SPP si NU ruleaza Base inainte/dupa.
echo.
echo Apasa orice tasta pentru a incepe...
pause >nul

echo.
echo [*] Pornire antrenare SPP-Ro-125M...
py run_scale_training.py --model spp
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [EROARE] Antrenarea modelului SPP a esuat cu codul %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================================
echo   [SUCCES] Antrenarea SPP-Ro-125M s-a finalizat cu succes!
echo ============================================================================
pause
