@echo off
chcp 65001 >nul
cd /d "%~dp0"
title SPP-Ro MT re-warm (SPP{MT} v1 + control)
if not exist logs mkdir logs
set LOG=logs\mt_rewarm.log

REM ---------------------------------------------------------------------------
REM  run_mt_rewarm.bat   SPP{MT} v1 cu program LR nou + control Base re-warm + toate evaluarile
REM
REM  De ce: prima incercare MT (run_overnight.bat, 2 nov) a reluat checkpoint-ul Base@45k CU optimizatorul
REM  si programul cosinus, deci cei 5.000 de pasi au rulat la LR 3.7e-6 -> 0; ponderile s-au miscat cu 0.03%
REM  si toate metricile au fost identice cu Base. Acel model este pastrat in models\spp_ro_88m_mt_cosinetail.
REM
REM  Aici: --init-from incarca DOAR ponderile Base@45k si porneste un program LR nou:
REM  peak 5e-5 (1/3 din peak-ul preantrenarii), warmup 250, cosinus -> 0 pe 5.000 pasi.
REM    [1/3] SPP{MT} v1   : 10% sidecar v1 sintetic           -> models\spp_ro_88m_mt        (~6 h)
REM    [2/3] control      : acelasi program, FARA reflectii   -> models\base_ro_88m_rewarm   (~6 h)
REM    [3/3] evaluari pentru toate modelele (~1 h)
REM  Reluabil: daca exista checkpoint-uri, reia de acolo; sare peste modelele terminate.
REM  Raport de progres la 30 min; checkpoint la fiecare 1.000 pasi. Live: watch_log.bat logs\mt_rewarm.log
REM ---------------------------------------------------------------------------

set MT_FLAGS=--init-from models/base_ro_125m/checkpoints/checkpoint-45000 --steps 5000 --total-target 50000 --lr 5e-5 --warmup-steps 250 --save-steps 1000 --interval-mins 30 --log-file %LOG%

echo ============================================================================
echo   SPP-Ro MT RE-WARM  started %date% %time%
echo   [1/3] SPP{MT} v1 (~6 h)  [2/3] control Base re-warm (~6 h)  [3/3] evaluari (~1 h)
echo   Log: %LOG%
echo ============================================================================
echo ===== MT REWARM START %date% %time% >> %LOG%

if not exist models\base_ro_125m\checkpoints\checkpoint-45000\model.safetensors (
    echo [ERROR] Lipseste models\base_ro_125m\checkpoints\checkpoint-45000\model.safetensors
    goto :err
)
if exist models\spp_ro_88m_mt\model.safetensors if not exist models\spp_ro_88m_mt_cosinetail (
    echo [ERROR] models\spp_ro_88m_mt exista deja si modelul vechi nu a fost mutat in models\spp_ro_88m_mt_cosinetail.
    echo         Daca acesta este deja modelul re-warm, sterge aceasta verificare. Altfel muta folderul vechi.
    goto :err
)

call :stamp "[1/3] SPP{MT} v1 - Base@45k + 5.000 pasi, 10%% sidecar v1, program LR nou"
if not exist models\spp_ro_88m_mt\model.safetensors (
    if exist models\spp_ro_88m_mt\checkpoints\checkpoint-* (
        echo        Reiau din ultimul checkpoint ...
        py src/train_scale_pretrain.py --mode spp %MT_FLAGS% --resume --output-dir models/spp_ro_88m_mt --repo-id flaviussteff/spp-ro-88m-mt
    ) else (
        py src/train_scale_pretrain.py --mode spp %MT_FLAGS% --output-dir models/spp_ro_88m_mt --repo-id flaviussteff/spp-ro-88m-mt
    )
    if errorlevel 1 goto :err
) else (
    echo        Modelul SPP{MT} exista, sar.
)

call :stamp "[2/3] Control Base re-warm - Base@45k + 5.000 pasi, acelasi program, fara reflectii"
if not exist models\base_ro_88m_rewarm\model.safetensors (
    if exist models\base_ro_88m_rewarm\checkpoints\checkpoint-* (
        echo        Reiau din ultimul checkpoint ...
        py src/train_scale_pretrain.py --mode base %MT_FLAGS% --resume --output-dir models/base_ro_88m_rewarm --repo-id flaviussteff/base-ro-88m-rewarm
    ) else (
        py src/train_scale_pretrain.py --mode base %MT_FLAGS% --output-dir models/base_ro_88m_rewarm --repo-id flaviussteff/base-ro-88m-rewarm
    )
    if errorlevel 1 goto :err
) else (
    echo        Controlul re-warm exista, sar.
)

if not exist evals\training_states mkdir evals\training_states
for %%M in (spp_ro_88m_mt base_ro_88m_rewarm) do (
    if exist models\%%M\checkpoints\checkpoint-5000\trainer_state.json copy /Y models\%%M\checkpoints\checkpoint-5000\trainer_state.json evals\training_states\trainer_state_%%M.json >nul
)

call :stamp "[3/3] Evaluari pentru toate modelele prezente (~1 h)"
call :runeval src\eval_constitution_v3.py
call :runeval src\eval_alignment_tax.py
call :runeval "src\eval_capabilities.py --n 1000"
call :runeval src\eval_jailbreaks.py

echo.
echo ============================================================================
echo   DONE %date% %time%  - rezultate in evals\ , modele in models\ , log %LOG%
echo ============================================================================
echo ===== MT REWARM DONE %date% %time% >> %LOG%
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

:err
echo.
echo [ERROR] %date% %time% Un pas a esuat. Ultimele linii din %LOG%:
echo ===== ERROR %date% %time% >> %LOG%
powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 40"
pause
exit /b 1
