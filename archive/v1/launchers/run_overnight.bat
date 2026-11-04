@echo off
chcp 65001 >nul
cd /d "%~dp0"
title SPP-Ro overnight: SPP{MT} baseline + corrected evaluations
if not exist logs mkdir logs
set LOG=logs\overnight.log

echo ============================================================================ >> %LOG%
echo  SPP-Ro OVERNIGHT RUN started %date% %time% >> %LOG%
echo ============================================================================ >> %LOG%

echo.
echo ============================================================================
echo   SPP-Ro OVERNIGHT RUN
echo   1. Build verified held-out set (FineWeb-2 shard 100, disjoint from training)
echo   2. Train SPP{MT}: Base checkpoint-45000 + 5,000 SPP steps (late insertion)
echo      output: models\spp_ro_88m_mt   (about 5 h on RTX 3060)
echo   3. Re-run all evaluations with the corrected scorer, incl. the MT model
echo   Log: %LOG%
echo ============================================================================
echo.

echo [1/6] Building held-out evaluation set ...
echo [1/6] build_heldout_set >> %LOG%
if not exist data\eval\heldout_fineweb.parquet (
    py src/build_heldout_set.py --shard 100 --n 3000 >> %LOG% 2>&1
    if errorlevel 1 goto :err
) else (
    echo        already exists, skipping. >> %LOG%
)

echo [2/6] Preparing SPP{MT} initialisation from Base checkpoint-45000 ...
echo [2/6] init MT >> %LOG%
if exist models\spp_ro_88m_mt if not exist models\spp_ro_88m_mt\NUL (
    echo [ERROR] models\spp_ro_88m_mt exists as a FILE, not a folder. Delete it and re-run. >> %LOG%
    goto :err
)
if not exist models\spp_ro_88m_mt\checkpoints\checkpoint-45000\model.safetensors (
    if not exist models\spp_ro_88m_mt\checkpoints mkdir models\spp_ro_88m_mt\checkpoints
    robocopy models\base_ro_125m\checkpoints\checkpoint-45000 models\spp_ro_88m_mt\checkpoints\checkpoint-45000 /E /R:2 /W:5 /NFL /NDL /NJH /NJS >> %LOG%
    if errorlevel 8 goto :err
    if not exist models\spp_ro_88m_mt\checkpoints\checkpoint-45000\optimizer.pt (
        echo [ERROR] checkpoint copy incomplete: optimizer.pt missing >> %LOG%
        goto :err
    )
)

echo [3/6] Training SPP{MT} (resume from step 45,000 to 50,000 with 10%% reflections) ...
echo [3/6] train MT >> %LOG%
if not exist models\spp_ro_88m_mt\model.safetensors (
    py src/train_scale_pretrain.py --mode spp --spp-ratio 0.10 --steps 50000 --resume --output-dir models/spp_ro_88m_mt --repo-id flaviussteff/spp-ro-88m-mt --interval-mins 30 --save-steps 2500 >> %LOG% 2>&1
    if errorlevel 1 goto :err
) else (
    echo        final MT model already exists, skipping training. >> %LOG%
)

echo [4/6] Archiving training states ...
if not exist evals\training_states mkdir evals\training_states
copy /Y models\spp_ro_88m_mt\checkpoints\checkpoint-50000\trainer_state.json evals\training_states\trainer_state_spp_mt.json >nul 2>&1
copy /Y models\base_ro_125m\checkpoints\checkpoint-50000\trainer_state.json evals\training_states\trainer_state_base.json >nul 2>&1
copy /Y models\spp_ro_125m\checkpoints\checkpoint-50000\trainer_state.json evals\training_states\trainer_state_spp10.json >nul 2>&1
copy /Y models\spp_ro_125m_20pct\checkpoints\checkpoint-50000\trainer_state.json evals\training_states\trainer_state_spp20.json >nul 2>&1

echo [5/6] Running corrected evaluations (ConstitutionEval-Ro, AIRisk-Ro, alignment tax on held-out, 600-item benchmark, adversarial prefixes) ...
echo [5/6] evals >> %LOG%
py src/eval_constitution.py >> %LOG% 2>&1
if errorlevel 1 echo [WARN] eval_constitution failed, continuing >> %LOG%
py src/eval_alignment_tax.py >> %LOG% 2>&1
if errorlevel 1 echo [WARN] eval_alignment_tax failed, continuing >> %LOG%
py src/build_master_benchmark_600.py >> %LOG% 2>&1
if errorlevel 1 echo [WARN] build_master_benchmark_600 failed, continuing >> %LOG%
py src/eval_jailbreaks.py >> %LOG% 2>&1
if errorlevel 1 echo [WARN] eval_jailbreaks failed, continuing >> %LOG%

echo [6/6] Done.
echo  FINISHED %date% %time% >> %LOG%
echo.
echo ============================================================================
echo   DONE. Results: evals\*.json, evals\*.tex ; model: models\spp_ro_88m_mt
echo   Check %LOG% for warnings.
echo ============================================================================
pause
exit /b 0

:err
echo.
echo [ERROR] A step failed. See %LOG% (last lines below):
powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 40"
pause
exit /b 1
