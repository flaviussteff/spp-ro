@echo off
chcp 65001 >nul
cd /d "%~dp0"
title SPP-Ro v2 pipeline
if not exist logs mkdir logs
set LOG=logs\v2_pipeline.log
set MODE=%1

REM ---------------------------------------------------------------------------
REM  run_v2_pipeline.bat            full: data -> SPP{T0} v2 (50 h) -> SPP{MT} v2 (5 h) -> Filtered (50 h) -> evals (1 h)
REM  run_v2_pipeline.bat nofiltered skip the Filtered baseline
REM  run_v2_pipeline.bat evalonly   only the evaluations
REM
REM  Progress: training prints a report every 30 minutes (step, ETA, finish time, loss, VRAM, sample text)
REM  and a line at every checkpoint (every 2,500 steps). Everything also goes to logs\v2_pipeline.log.
REM  Resumable: re-running skips finished models and resumes training from the last checkpoint.
REM  Live view from another window: watch_log.bat logs\v2_pipeline.log
REM ---------------------------------------------------------------------------

set TRAIN_FLAGS=--mode spp --inline --sidecar data/sidecar/reflections_v2.parquet --mask-assistant-loss --snap-insertion --prefix-tail --interval-mins 30 --save-steps 2500 --log-file %LOG%
REM MT: weights from Base@45k, fresh LR schedule (continuing the cosine tail gives LR ~1e-6 and leaves the weights unchanged - see Errata 9)
set MT_FLAGS=--init-from models/base_ro_125m/checkpoints/checkpoint-45000 --steps 5000 --total-target 50000 --lr 5e-5 --warmup-steps 250 --save-steps 1000
REM MT v2 data recipe as in the paper's midtraining (EPFL run 5_spp_mt): annotated documents up-weighted to 67.6%% of the
REM sequences (one annotated document per sequence), loss only on the reflection span of annotated documents; plain documents keep the full loss.
set MT_SPP_FLAGS=--mt-annotated-ratio 0.676 --reflection-loss-only

call :banner "SPP-Ro V2 PIPELINE  (mode: %MODE%)  started %date% %time%"
echo   Plan si durate estimate pe RTX 3060 (4.3 s/pas):
echo     [1/6] date v2 (API, fara GPU)            : ore, depinde de API
echo     [2/6] SPP{T0} v2, 50.000 pasi de la zero  : ~60 h
echo     [3/6] SPP{MT} v2, 5.000 pasi din Base@45k : ~6 h  (+6 h control re-warm daca lipseste)
echo     [4/6] Filtered, 50.000 pasi de la zero    : ~60 h   (sarit cu "nofiltered")
echo     [5/6] arhivare trainer_state
echo     [6/6] evaluari (toate modelele)           : ~1 h
echo   Raport de progres la fiecare 30 min; checkpoint la fiecare 2.500 pasi (ultimele 2 pastrate).
echo   Log complet: %LOG%
echo.
echo ===== PIPELINE START %date% %time% mode=%MODE% >> %LOG%

if /I "%MODE%"=="evalonly" goto :evals

call :stamp "[1/6] Held-out set + reflections v2"
if not exist data\eval\heldout_fineweb.parquet (
    py src/build_heldout_set.py --shard 100 --n 3000
    if errorlevel 1 goto :err
)
if not exist data\sidecar\reflections_v2.parquet (
    echo        Datele v2 lipsesc: rulez calea B (etichete Qwen -^> clasificator BERT -^> adnotare Qwen) ...
    call run_generate_v2_data.bat all
    if errorlevel 1 goto :err
) else (
    echo        reflections_v2.parquet exista, sar peste generare.
)
py -X utf8 -c "import pandas as pd,sys; d=pd.read_parquet('data/sidecar/reflections_v2.parquet'); n=len(d); h=int((d['severity']>=3).sum()); print(f'reflections_v2: {n} documente, {h} cu severitate>=3'); sys.exit(0 if n>=2000 else 3)"
if errorlevel 3 (
    echo [ERROR] reflections_v2.parquet are sub 2000 documente; generarea este incompleta. Ruleaza din nou generate_reflections_v2.py.
    goto :err
)

call :stamp "[2/6] SPP{T0} v2 - 50.000 pasi de la zero (~60 h)"
if not exist models\spp_ro_88m_v2\model.safetensors (
    if exist models\spp_ro_88m_v2\checkpoints\checkpoint-* (
        echo        Reiau din ultimul checkpoint ...
        py src/train_scale_pretrain.py %TRAIN_FLAGS% --steps 50000 --resume --output-dir models/spp_ro_88m_v2 --repo-id flaviussteff/spp-ro-88m-v2
    ) else (
        py src/train_scale_pretrain.py %TRAIN_FLAGS% --steps 50000 --from-scratch --output-dir models/spp_ro_88m_v2 --repo-id flaviussteff/spp-ro-88m-v2
    )
    if errorlevel 1 goto :err
) else (
    echo        Modelul T0 v2 exista, sar.
)

call :stamp "[3/6] SPP{MT} v2 - Base@45k + 5.000 pasi pe fluxul v2, program LR nou (peak 5e-5, warmup 250) (~6 h)"
if exist models\spp_ro_88m_mt_v2 if not exist models\spp_ro_88m_mt_v2\NUL (
    echo [ERROR] models\spp_ro_88m_mt_v2 exista ca FISIER. Sterge-l.
    goto :err
)
if not exist models\spp_ro_88m_mt_v2\model.safetensors (
    if exist models\spp_ro_88m_mt_v2\checkpoints\checkpoint-* (
        py src/train_scale_pretrain.py %TRAIN_FLAGS% %MT_FLAGS% %MT_SPP_FLAGS% --resume --output-dir models/spp_ro_88m_mt_v2 --repo-id flaviussteff/spp-ro-88m-mt-v2
    ) else (
        py src/train_scale_pretrain.py %TRAIN_FLAGS% %MT_FLAGS% %MT_SPP_FLAGS% --output-dir models/spp_ro_88m_mt_v2 --repo-id flaviussteff/spp-ro-88m-mt-v2
    )
    if errorlevel 1 goto :err
) else (
    echo        Modelul MT v2 exista, sar.
)
if not exist models\base_ro_88m_rewarm\model.safetensors (
    echo        Controlul re-warm lipseste ^(exista deja din 3 nov; launcher-ul vechi e in archive\v1\launchers^); il antrenez acum ^(~6 h^) ...
    if exist models\base_ro_88m_rewarm\checkpoints\checkpoint-* (
        py src/train_scale_pretrain.py --mode base %MT_FLAGS% --resume --interval-mins 30 --log-file %LOG% --output-dir models/base_ro_88m_rewarm --repo-id flaviussteff/base-ro-88m-rewarm
    ) else (
        py src/train_scale_pretrain.py --mode base %MT_FLAGS% --interval-mins 30 --log-file %LOG% --output-dir models/base_ro_88m_rewarm --repo-id flaviussteff/base-ro-88m-rewarm
    )
    if errorlevel 1 goto :err
)

if /I "%MODE%"=="nofiltered" goto :states
call :stamp "[4/6] Filtered baseline - 50.000 pasi de la zero, documente nocive loss-masked (~60 h)"
if not exist models\filtered_ro_88m\model.safetensors (
    if exist models\filtered_ro_88m\checkpoints\checkpoint-* (
        py src/train_scale_pretrain.py --mode filtered --sidecar data/sidecar/reflections_v2.parquet --interval-mins 30 --save-steps 2500 --log-file %LOG% --steps 50000 --resume --output-dir models/filtered_ro_88m --repo-id flaviussteff/filtered-ro-88m
    ) else (
        py src/train_scale_pretrain.py --mode filtered --sidecar data/sidecar/reflections_v2.parquet --interval-mins 30 --save-steps 2500 --log-file %LOG% --steps 50000 --from-scratch --output-dir models/filtered_ro_88m --repo-id flaviussteff/filtered-ro-88m
    )
    if errorlevel 1 goto :err
) else (
    echo        Modelul Filtered exista, sar.
)

:states
call :stamp "[5/6] Arhivez trainer_state.json"
if not exist evals\training_states mkdir evals\training_states
for %%M in (base_ro_125m spp_ro_125m spp_ro_125m_20pct spp_ro_88m_mt base_ro_88m_rewarm spp_ro_88m_v2 spp_ro_88m_mt_v2 filtered_ro_88m) do (
    if exist models\%%M\checkpoints\checkpoint-50000\trainer_state.json copy /Y models\%%M\checkpoints\checkpoint-50000\trainer_state.json evals\training_states\trainer_state_%%M.json >nul
    if exist models\%%M\checkpoints\checkpoint-5000\trainer_state.json copy /Y models\%%M\checkpoints\checkpoint-5000\trainer_state.json evals\training_states\trainer_state_%%M.json >nul
)

:evals
call :stamp "[6/6] Evaluari pentru toate modelele prezente (~1 h): ConstitutionEval-Ro v3, PPL held-out, cloze, prefixe adversariale"
call :runeval src\eval_constitution_v3.py
call :runeval src\eval_alignment_tax.py
call :runeval "src\eval_capabilities.py --n 1000"
call :runeval src\eval_jailbreaks.py

call :banner "DONE %date% %time%  - rezultate in evals\ , modele in models\ , log %LOG%"
echo ===== PIPELINE DONE %date% %time% >> %LOG%
pause
exit /b 0

:runeval
echo   %time%  ruleaza %~1 ...
echo ----- %time% %~1 >> %LOG%
py %~1 >> %LOG% 2>&1
if errorlevel 1 (
    echo   [WARN] %~1 a esuat - vezi %LOG%
    echo [WARN] %~1 failed >> %LOG%
) else (
    echo   %time%  OK %~1
)
exit /b 0

:stamp
echo.
echo ---------------------------------------------------------------------------
echo  %date% %time%   %~1
echo ---------------------------------------------------------------------------
echo ----- %date% %time% %~1 >> %LOG%
exit /b 0

:banner
echo.
echo ============================================================================
echo   %~1
echo ============================================================================
exit /b 0

:err
echo.
echo [ERROR] %date% %time% Un pas a esuat. Ultimele linii din %LOG%:
echo ===== ERROR %date% %time% >> %LOG%
powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 40"
pause
exit /b 1
