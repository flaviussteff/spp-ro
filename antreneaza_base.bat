@echo off
cd /d "%~dp0"
title Base-Ro-125M Scale Training

echo ============================================================================
echo   SPP-Ro: Antrenare Scalata Independenta - Model BASE (Base-Ro-125M)
echo   Hardware: NVIDIA GeForce RTX 3060 12GB (BF16)
echo   Reluare automata din ultimul checkpoint (ex. checkpoint-40000)
echo ============================================================================
echo.
echo Model: Base-Ro-125M (+50.000 pasi, max 60.000)
echo NOTA: Acest script antreneaza DOAR modelul Base si NU porneste SPP dupa.
echo.
echo Apasa orice tasta pentru a incepe...
pause >nul

echo.
echo [*] Pornire antrenare Base-Ro-125M...
py run_scale_training.py --model base
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [EROARE] Antrenarea modelului Base a esuat cu codul %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================================
echo   [SUCCES] Antrenarea Base-Ro-125M s-a finalizat cu succes!
echo ============================================================================
pause
