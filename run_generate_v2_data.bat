@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title SPP-Ro: date v2, calea B (etichete -> clasificator -> adnotare)
if not exist logs mkdir logs
set LOG=logs\generate_v2.log

REM ---------------------------------------------------------------------------
REM  Calea B (ca in articol): 
REM   [1/3] Qwen eticheteaza severitatea 0-5 pentru ~3000 documente (2/3 candidati lexicali, 1/3 aleatoare)  ~ $2, 10 min, fara GPU
REM   [2/3] un BERT romanesc este antrenat pe etichete si scoreaza TOT corpusul (1,5M documente)            ~ 10 min + 1-2 h GPU
REM         -> data\clean\spp_candidates_v2.parquet = toate documentele semnalate nocive + tot atatea benigne
REM   [3/3] Qwen scrie reflectiile pentru exact aceste documente                                            ~ API, 1-3 h, costul principal
REM  Chei: .env, o linie: CEREBRAS_API_KEYS=csk-...,csk-...   (toate in paralel; cele fara credit sunt retrase automat)
REM  Reluabil: fiecare pas sare daca iesirea lui exista; pasul 3 continua de unde a ramas.
REM  Usage: run_generate_v2_data.bat            (toti pasii, Qwen)
REM         run_generate_v2_data.bat labels     (doar pasul 1 - afiseaza si estimarea de cost pentru pasul 3)
REM         run_generate_v2_data.bat classify   (doar pasul 2; NU rula cat timp alt antrenament ocupa GPU-ul)
REM         run_generate_v2_data.bat annotate   (doar pasul 3)
REM ---------------------------------------------------------------------------
set STEP=%1
set MODEL=qwen-3.8-27b

echo ============================================================================
echo   DATE v2 - CALEA B   (pas: %STEP%)   model anotator: %MODEL%
echo   log: %LOG%
echo ============================================================================
if not exist .env (
    echo [ERROR] Lipseste .env cu CEREBRAS_API_KEYS=...
    pause & exit /b 1
)
py -X utf8 -c "import re,sys; s=open('.env',encoding='utf-8').read(); m=re.search(r'CEREBRAS_API_KEYS=([^\r\n]*)',s); ks=[k.strip() for k in (m.group(1) if m else '').split(',') if k.strip()]; print(f'Chei in .env: {len(ks)}'); sys.exit(0 if ks else 2)"
if errorlevel 2 (
    echo [ERROR] Nicio cheie in .env.
    pause & exit /b 1
)
echo ===== START %date% %time% step=%STEP% >> %LOG%

if /I "%STEP%"=="classify" goto :classify
if /I "%STEP%"=="annotate" goto :annotate

:labels
echo.
echo [1/3] %time% Etichete de severitate (Qwen, ~3000 documente) ...
if exist data\sidecar\severity_labels.parquet (
    echo        severity_labels.parquet exista, sar.
) else (
    py src/generate_reflections_v2.py --label-only --n-label 3000 --model %MODEL%
    if errorlevel 1 goto :err
)
if /I "%STEP%"=="labels" goto :end

:classify
echo.
echo [2/3] %time% Clasificator BERT + scorarea corpusului (GPU) ...
if exist data\clean\spp_candidates_v2.parquet (
    echo        spp_candidates_v2.parquet exista, sar.
) else (
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > logs\gpu_mem.tmp 2>nul
    set /p GPUMEM=<logs\gpu_mem.tmp
    if defined GPUMEM if !GPUMEM! GTR 3000 (
        echo [ATENTIE] GPU-ul are deja !GPUMEM! MiB ocupati - probabil ruleaza un antrenament.
        echo           Pasul 2 va fi lent si poate da out-of-memory. Apasa o tasta pentru a continua oricum sau inchide fereastra.
        pause >nul
    )
    py src/train_safety_classifier.py
    if errorlevel 1 goto :err
)
if /I "%STEP%"=="classify" goto :end

:annotate
echo.
echo [3/3] %time% Reflectii pentru documentele semnalate (Qwen) ...
if not exist data\clean\spp_candidates_v2.parquet (
    echo [ERROR] Lipseste data\clean\spp_candidates_v2.parquet - ruleaza pasul 2.
    goto :err
)
py src/generate_reflections_v2.py --candidates data/clean/spp_candidates_v2.parquet --batch 3 --model %MODEL%
if errorlevel 1 goto :err

:end
echo ===== END %date% %time% >> %LOG%
echo.
if exist data\sidecar\reflections_v2_progress.json type data\sidecar\reflections_v2_progress.json
echo.
echo Daca scrie "Toate cheile sunt epuizate", pune chei noi in .env si ruleaza din nou (continua de unde a ramas).
pause
exit /b 0

:err
echo [ERROR] %date% %time% pasul a esuat.
echo ===== ERROR %date% %time% >> %LOG%
pause
exit /b 1
