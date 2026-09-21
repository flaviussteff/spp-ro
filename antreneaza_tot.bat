@echo off
cd /d "%~dp0"
title SPP-Ro Scale Training (Secvential: Base + SPP)

echo ============================================================================
echo   SPP-Ro: Antrenare Secventiala Completa (Base urmat de SPP)
echo   Daca doresti sa le rulezi separat, foloseste:
echo     - antreneaza_base.bat (doar Base-Ro-125M)
echo     - antreneaza_spp.bat  (doar SPP-Ro-125M)
echo ============================================================================
echo.
echo Modele in secventa:
echo   1. Base-Ro-125M (+50.000 pasi, max 60.000)
echo   2. SPP-Ro-125M  (+50.000 pasi, max 60.000)
echo.
echo Apasa orice tasta pentru a incepe ambele etape secvential...
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
echo ============================================================================
echo   Antrenare secventiala finalizata cu succes pentru ambele modele!
echo ============================================================================
pause
