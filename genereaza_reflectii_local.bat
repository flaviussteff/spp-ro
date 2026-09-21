@echo off
cd /d "%~dp0"
title SPP-Ro Local GPU Reflection Generator (Qwen 2.5 7B)

echo ============================================================================
echo   SPP-Ro: Generator Local de Reflexii Constitutionale (60.000 Reflexii)
echo   Model Teacher: Qwen/Qwen2.5-7B-Instruct (4-bit BitsAndBytes NF4)
echo   Hardware: NVIDIA GeForce RTX 3060 12GB (Consum VRAM: ~5.5 GB)
echo   Limite Token-uri: FARA LIMITA (100% Offline / Local pe GPU)
echo   Inspectie Live: Afisare text complet + taguri ^<assistant^> din 1.000 in 1.000
echo   Salvare Securizata: Atomic swap + backup rotativ (.bak)
echo ============================================================================
echo.
echo NOTA: Asigura-te ca antrenarea Base-Ro s-a incheiat pentru ca GPU-ul sa fie liber.
echo.
echo Apasa orice tasta pentru a porni generarea automata cu Qwen 2.5 7B...
pause >nul

echo.
echo [*] Pornire generare cu Qwen 2.5 7B Instruct pe GPU...
py src\generate_local_llm_reflections.py --model Qwen/Qwen2.5-7B-Instruct --batch-size 4 --interval 1000
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [EROARE] Generarea s-a oprit cu codul %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================================
echo   [SUCCES] Toate cele 60.000 de reflexii au fost generate cu succes!
echo ============================================================================
pause
