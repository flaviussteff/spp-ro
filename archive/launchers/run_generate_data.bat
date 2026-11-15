@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title SPP-Ro: data (severity labels -> safety classifier -> reflections)
if not exist logs mkdir logs
set LOG=logs\generate_data.log

REM ---------------------------------------------------------------------------
REM  The paper's data path B, in three steps:
REM   [1/3] Qwen labels the severity 0-5 of ~3,000 documents (2/3 lexical candidates, 1/3 random)        ~ $2, 10 min, no GPU
REM   [2/3] a Romanian BERT is trained on the labels and scores the WHOLE corpus (1.5M documents)         ~ 10 min + 1-2 h GPU
REM         -> data\clean\spp_candidates.parquet = every document flagged as harmful + as many benign ones
REM   [3/3] Qwen writes the reflections for exactly these documents                                        ~ API, 1-3 h, the main cost
REM         -> data\sidecar\reflections_first_pass.parquet (run_expand_annotations.bat then expands it to 10 percent)
REM  Keys: .env, one line: CEREBRAS_API_KEYS=csk-...,csk-...   (all in parallel; keys without credit are retired automatically)
REM  Resumable: every step is skipped if its output exists; step 3 continues where it stopped.
REM  Usage: run_generate_data.bat            (all steps)
REM         run_generate_data.bat labels     (step 1 only - also prints the cost estimate for step 3)
REM         run_generate_data.bat classify   (step 2 only; do NOT run while another training uses the GPU)
REM         run_generate_data.bat annotate   (step 3 only)
REM ---------------------------------------------------------------------------
set STEP=%1
set MODEL=qwen-3.8-27b

echo ============================================================================
echo   DATA - PATH B   (step: %STEP%)   annotator model: %MODEL%
echo   log: %LOG%
echo ============================================================================
if not exist .env (
    echo [ERROR] .env with CEREBRAS_API_KEYS=... is missing
    pause & exit /b 1
)
py -X utf8 -c "import re,sys; s=open('.env',encoding='utf-8').read(); m=re.search(r'CEREBRAS_API_KEYS=([^\r\n]*)',s); ks=[k.strip() for k in (m.group(1) if m else '').split(',') if k.strip()]; print(f'Keys in .env: {len(ks)}'); sys.exit(0 if ks else 2)"
if errorlevel 2 (
    echo [ERROR] No key in .env.
    pause & exit /b 1
)
echo ===== START %date% %time% step=%STEP% >> %LOG%

if /I "%STEP%"=="classify" goto :classify
if /I "%STEP%"=="annotate" goto :annotate

:labels
echo.
echo [1/3] %time% Severity labels (Qwen, ~3,000 documents) ...
if exist data\sidecar\severity_labels.parquet (
    echo        severity_labels.parquet exists, skipping.
) else (
    py src/generate_reflections.py --label-only --n-label 3000 --model %MODEL%
    if errorlevel 1 goto :err
)
if /I "%STEP%"=="labels" goto :end

:classify
echo.
echo [2/3] %time% BERT classifier + corpus scoring (GPU) ...
if exist data\clean\spp_candidates.parquet (
    echo        spp_candidates.parquet exists, skipping.
) else (
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > logs\gpu_mem.tmp 2>nul
    set /p GPUMEM=<logs\gpu_mem.tmp
    if defined GPUMEM if !GPUMEM! GTR 3000 (
        echo [WARNING] The GPU already has !GPUMEM! MiB in use - a training is probably running.
        echo           Step 2 will be slow and may run out of memory. Press a key to continue anyway or close this window.
        pause >nul
    )
    py src/train_safety_classifier.py
    if errorlevel 1 goto :err
)
if /I "%STEP%"=="classify" goto :end

:annotate
echo.
echo [3/3] %time% Reflections for the selected documents (Qwen) ...
if not exist data\clean\spp_candidates.parquet (
    echo [ERROR] data\clean\spp_candidates.parquet is missing - run step 2.
    goto :err
)
py src/generate_reflections.py --candidates data/clean/spp_candidates.parquet --batch 3 --model %MODEL%
if errorlevel 1 goto :err

:end
echo ===== END %date% %time% >> %LOG%
echo.
if exist data\sidecar\reflections_first_pass_progress.json type data\sidecar\reflections_first_pass_progress.json
echo.
echo If it says that all keys are exhausted, put new keys in .env and run again (it continues where it stopped).
pause
exit /b 0

:err
echo [ERROR] %date% %time% the step failed.
echo ===== ERROR %date% %time% >> %LOG%
pause
exit /b 1
