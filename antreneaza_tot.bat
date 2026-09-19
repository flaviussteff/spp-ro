@echo off
cd /d "%~dp0"
title SPP-Ro Training

echo SPP-Ro: Antrenare Scalata (Base + SPP)
echo.
echo Modele:
echo   1. Base-Ro-125M (+50.000 pasi, ~50h)
echo   2. SPP-Ro-125M  (+50.000 pasi, ~50h)
echo.
echo Apasa orice tasta pentru a incepe...
pause >nul

echo.
echo [1/2] Rulare Base-Ro-125M...
py run_scale_training.py --model base
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Eroare] Antrenarea modelului Base a esuat cu codul %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/2] Rulare SPP-Ro-125M...
py run_scale_training.py --model spp
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Eroare] Antrenarea modelului SPP a esuat cu codul %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Antrenare finalizata cu succes.
pause
