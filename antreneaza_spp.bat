@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Antrenare SPP-Ro-125M (Token Zero - 50.000 Pasi - Compute-Matched cu Base-Ro)

echo ============================================================================
echo   SPP-Ro: Antrenare din Token Zero - Model CONSTITUTIONAL (SPP-Ro-125M)
echo   Hardware: NVIDIA GeForce RTX 3060 12GB (BF16 / SDPA)
echo   Dataset: 60.000 reflexii constitutionale (50%% Sensibile + 50%% Factuale)
echo   Rata Interleaving: 10%% SPP (Reflectii) / 90%% Text Liber
echo   Configuratie: 50.000 PASI (3.200.000 texte, ~3.27B tokeni) - IDENTIC BASE-RO
echo   Durata estimata: ~48-50 ore (salvare automata la fiecare 5.000 de pasi)
echo   Dupa antrenare: Publicare automata pe Hugging Face (flaviussteff/spp-ro-125m)
echo ============================================================================
echo.
echo Modelul se va antrena complet de la pasul zero (Token Zero), in conditii
echo identice cu Base-Ro-125M pentru o comparatie stiintifica de nota 10 in licenta.
echo.
echo Apasa orice tasta pentru a incepe antrenarea...
pause >nul

echo.
echo [*] Pornire antrenare SPP-Ro-125M (50.000 pasi, rapoarte orare live)...
py src/train_scale_pretrain.py --mode spp --steps 50000 --from-scratch --push-to-hub --interval-mins 60 --save-steps 5000
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [EROARE] Antrenarea modelului SPP a esuat cu codul %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================================
echo   [SUCCES] Antrenarea celor 50.000 de pasi si publicarea pe Hugging Face s-au incheiat!
echo   Modelul este disponibil la: https://huggingface.co/flaviussteff/spp-ro-125m
echo ============================================================================
pause
