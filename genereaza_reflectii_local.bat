@echo off
cd /d "%~dp0"
title SPP-Ro Local GPU Reflection Generator (Qwen 2.5 7B)

echo ============================================================================
echo   SPP-Ro: Generator Local de Reflexii Constitutionale (6.000 Reflexii)
echo   Model Teacher: Qwen/Qwen2.5-7B-Instruct (4-bit BitsAndBytes NF4)
echo   Hardware: NVIDIA GeForce RTX 3060 12GB (Batch Size: 8)
echo   Inspectie Live: Afisare text complet + taguri ^<assistant^> din 500 in 500
echo   Salvare Securizata: Atomic swap + backup rotativ (.bak)
echo ============================================================================
echo.
echo NOTA: Asigura-te ca antrenarea Base-Ro s-a incheiat pentru ca GPU-ul sa fie liber.
echo.
echo Apasa orice tasta pentru a porni generarea automata cu Qwen 2.5 7B...
pause >nul

echo.
echo [*] Pornire generare cu Qwen 2.5 7B Instruct pe GPU...
py src\generate_local_llm_reflections.py --model Qwen/Qwen2.5-7B-Instruct --target 6000 --batch-size 8 --max-new-tokens 360 --interval 500
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [EROARE] Generarea s-a oprit cu codul %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================================
echo   [SUCCES] Toate cele 6.000 de reflexii au fost generate cu succes!
echo ============================================================================
pause
