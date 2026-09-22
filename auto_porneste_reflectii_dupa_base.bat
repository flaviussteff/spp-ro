@echo off
cd /d "%~dp0"
title SPP-Ro: Monitor si Auto-Pornire Generare Reflexii Qwen 2.5 7B

echo ============================================================================
echo   SPP-Ro: Monitor si Auto-Pornire Generare Reflexii Qwen 2.5 7B pe GPU
echo ============================================================================
echo.
echo Acest script monitorizeaza automat antrenarea Base-Ro (pasul 50.000).
echo Imediat ce antrenarea se termina si memoria GPU este eliberata, va porni
echo automat generarea celor 60.000 de reflexii cu Qwen 2.5 7B Instruct (4-bit).
echo.
echo Poti lasa aceasta fereastra deschisa peste noapte (poti inchide Chrome / Antigravity).
echo.
echo [*] Pornire monitorizare automata...
echo.

py src\auto_chain_reflections.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [EROARE] Scriptul de monitorizare/generare s-a oprit cu codul %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================================
echo   [SUCCES] Procesul complet s-a finalizat!
echo ============================================================================
pause
