@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title SPP-Ro v3 pipeline (packing)
if not exist logs mkdir logs
set LOG=logs\v3_pipeline.log

REM ---------------------------------------------------------------------------
REM  Generatia 3 = reteta v2 + packing (documente terminate cu </s> umplu fiecare fereastra de 1.024 pozitii,
REM  atentie bloc-diagonala pe document -> ~2x tokeni reali pe pas la acelasi cost) + sidecar extins la ~10 la suta din documente.
REM
REM  run_v3_pipeline.bat                  tot: candidati v3 (GPU ~1 h) -> adnotare (API, fereastra separata, in paralel)
REM                                       -> Base-pack (60 h) -> [asteapta adnotarea] -> SPP{T0}-pack (60 h)
REM                                       -> re-warm-pack (6 h) -> SPP{MT}-pack (6 h) -> Filtered-pack (60 h) -> evaluari (1 h)
REM  run_v3_pipeline.bat nofiltered       fara Filtered-pack
REM  run_v3_pipeline.bat noexpand         fara extinderea sidecar-ului (reflections_v3 = copie a lui reflections_v2)
REM  run_v3_pipeline.bat nofiltered noexpand
REM  run_v3_pipeline.bat evalonly         doar evaluarile
REM
REM  Opreste intai orice antrenament v2 (Ctrl+C in fereastra lui): pipeline-ul verifica VRAM-ul la pornire.
REM  Raport la 30 min, checkpoint la 2.500 pasi; tot ce se afiseaza merge si in logs\v3_pipeline.log.
REM  Reluabil: modelele terminate sunt sarite, antrenarile reiau din ultimul checkpoint (ponderi fp32, Adam fp32).
REM  Live din alta fereastra: watch_log.bat logs\v3_pipeline.log
REM ---------------------------------------------------------------------------
set SKIPFILT=0
set SKIPEXP=0
set EVALONLY=0
for %%A in (%1 %2) do (
    if /I "%%A"=="nofiltered" set SKIPFILT=1
    if /I "%%A"=="noexpand" set SKIPEXP=1
    if /I "%%A"=="evalonly" set EVALONLY=1
)
set CLF_THR=0.1
set SIDECAR=data\sidecar\reflections_v3.parquet

set COMMON=--pack --interval-mins 30 --save-steps 2500 --log-file %LOG%
set SPP_FLAGS=--mode spp --inline --sidecar data/sidecar/reflections_v3.parquet --mask-assistant-loss --snap-insertion --prefix-tail %COMMON%
set BASE_FLAGS=--mode base %COMMON%
set FILT_FLAGS=--mode filtered --sidecar data/sidecar/reflections_v3.parquet %COMMON%
REM MT / re-warm: weights from Base-pack@45k (kept by --keep-step), fresh LR schedule (Errata 9)
set MT_FLAGS=--init-from models/base_ro_88m_pack/checkpoints_keep/checkpoint-45000 --steps 5000 --total-target 50000 --lr 5e-5 --warmup-steps 250 --save-steps 1000
set MT_SPP_FLAGS=--mt-annotated-ratio 0.676 --reflection-loss-only

call :banner "SPP-Ro V3 PIPELINE (packing)  nofiltered=%SKIPFILT% noexpand=%SKIPEXP%  started %date% %time%"
echo ===== PIPELINE START %date% %time% args=%1 %2 >> %LOG%
if "%EVALONLY%"=="1" goto :evals

call :stamp "[0/9] Auto-test packing (CPU, ~20 s)"
py src\test_packing.py
if errorlevel 1 (
    echo [ERROR] Testul de packing a esuat - nu pornesc antrenarea.
    goto :err
)
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > logs\gpu_mem.tmp 2>nul
set /p GPUMEM=<logs\gpu_mem.tmp
if defined GPUMEM if !GPUMEM! GTR 3000 (
    echo [ATENTIE] GPU-ul are deja !GPUMEM! MiB ocupati - probabil ruleaza inca antrenamentul v2.
    echo           Opreste-l intai ^(Ctrl+C in fereastra lui^). Apasa o tasta pentru a continua oricum sau inchide fereastra.
    pause >nul
)

call :stamp "[1/9] Extinderea sidecar-ului la ~10 la suta din documente (prag clasificator %CLF_THR%)"
if "%SKIPEXP%"=="1" (
    echo        noexpand: reflections_v3.parquet va fi o copie a lui reflections_v2.parquet.
    goto :base
)
if exist data\sidecar\expand_v3.done (
    echo        expand_v3.done exista: extinderea este deja completa.
    goto :base
)
if not exist data\clean\spp_candidates_v3.parquet (
    echo        Scorez corpusul cu clasificatorul BERT la pragul %CLF_THR%, fara documentele deja adnotate ^(GPU, ~1 h^) ...
    py src/train_safety_classifier.py --score-only --threshold %CLF_THR% --max-flagged 0 --exclude-parquet data/sidecar/reflections_v2.parquet --out-candidates data/clean/spp_candidates_v3.parquet
    if errorlevel 1 goto :err
) else (
    echo        spp_candidates_v3.parquet exista, sar scorarea.
)
REM do not start a second annotation process if one is already running (re-launch of the pipeline)
tasklist /FI "WINDOWTITLE eq SPP-Ro: extindere sidecar v3*" 2>nul | find /I "cmd.exe" >nul
if not errorlevel 1 (
    echo        Fereastra de adnotare ruleaza deja, nu pornesc alta.
    goto :base
)
if exist data\sidecar\expand_v3.running (
    echo        [ATENTIE] data\sidecar\expand_v3.running exista dar nu vad fereastra de adnotare: o sterg si pornesc adnotarea din nou.
    del data\sidecar\expand_v3.running >nul 2>&1
)
echo        Pornesc adnotarea in fereastra separata "SPP-Ro: extindere sidecar v3" ^(doar API, ruleaza in paralel cu Base-pack^) ...
start "SPP-Ro: extindere sidecar v3" cmd /c run_expand_annotations.bat

:base
call :stamp "[2/9] Base-pack - 50.000 pasi de la zero, packing (~60 h)"
if not exist models\base_ro_88m_pack\model.safetensors (
    if exist models\base_ro_88m_pack\checkpoints\checkpoint-* (
        echo        Reiau din ultimul checkpoint ...
        py src/train_scale_pretrain.py %BASE_FLAGS% --steps 50000 --keep-step 45000 --resume --output-dir models/base_ro_88m_pack --repo-id flaviussteff/base-ro-88m-pack
    ) else (
        py src/train_scale_pretrain.py %BASE_FLAGS% --steps 50000 --keep-step 45000 --from-scratch --output-dir models/base_ro_88m_pack --repo-id flaviussteff/base-ro-88m-pack
    )
    if errorlevel 1 goto :err
) else (
    echo        Base-pack exista, sar.
)
if not exist models\base_ro_88m_pack\checkpoints_keep\checkpoint-45000\model.safetensors (
    echo [ERROR] Lipseste models\base_ro_88m_pack\checkpoints_keep\checkpoint-45000 ^(ar fi trebuit pastrat de --keep-step^); MT si re-warm nu pot porni.
    goto :err
)

call :stamp "[3/9] Sidecar-ul v3 (acelasi fisier pentru SPP-pack, MT-pack si Filtered-pack)"
if "%SKIPEXP%"=="1" (
    if not exist %SIDECAR% copy /Y data\sidecar\reflections_v2.parquet %SIDECAR% >nul
    goto :checksidecar
)
:waitann
if exist data\sidecar\expand_v3.done goto :checksidecar
echo   %time%  astept extinderea sidecar-ului ^(fereastra "SPP-Ro: extindere sidecar v3"^). Daca s-a oprit fara chei: adauga chei in .env si ruleaza run_expand_annotations.bat
timeout /t 600 /nobreak >nul
goto :waitann
:checksidecar
if not exist %SIDECAR% (
    echo [ERROR] Lipseste %SIDECAR%.
    goto :err
)
py -X utf8 -c "import pandas as pd,sys; d=pd.read_parquet('data/sidecar/reflections_v3.parquet'); n=len(d); print(f'reflections_v3.parquet: {n} documente ({100*n/1500000:.1f} la suta din corpus), severitate>=3: {int((d.severity>=3).sum())}, clf_flag=1: {int((d.clf_flag==1).sum())}, chei duplicate: {int(d.doc_key.duplicated().sum())}'); sys.exit(0 if (n>=2000 and not d.doc_key.duplicated().any()) else 3)"
if errorlevel 1 (
    echo [ERROR] reflections_v3.parquet este invalid ^(sub 2000 documente, chei duplicate sau fisier corupt^).
    goto :err
)

:spp
call :stamp "[4/9] SPP{T0}-pack - 50.000 pasi de la zero, packing (~60 h)"
if not exist models\spp_ro_88m_pack\model.safetensors (
    if exist models\spp_ro_88m_pack\checkpoints\checkpoint-* (
        echo        Reiau din ultimul checkpoint ...
        py src/train_scale_pretrain.py %SPP_FLAGS% --steps 50000 --resume --output-dir models/spp_ro_88m_pack --repo-id flaviussteff/spp-ro-88m-pack
    ) else (
        py src/train_scale_pretrain.py %SPP_FLAGS% --steps 50000 --from-scratch --output-dir models/spp_ro_88m_pack --repo-id flaviussteff/spp-ro-88m-pack
    )
    if errorlevel 1 goto :err
) else (
    echo        SPP-pack exista, sar.
)

call :stamp "[5/9] Control re-warm-pack: Base-pack@45k + 5.000 pasi fara reflectii (~6 h)"
if not exist models\base_ro_88m_pack_rewarm\model.safetensors (
    if exist models\base_ro_88m_pack_rewarm\checkpoints\checkpoint-* (
        py src/train_scale_pretrain.py --mode base --pack %MT_FLAGS% --resume --interval-mins 30 --log-file %LOG% --output-dir models/base_ro_88m_pack_rewarm --repo-id flaviussteff/base-ro-88m-pack-rewarm
    ) else (
        py src/train_scale_pretrain.py --mode base --pack %MT_FLAGS% --interval-mins 30 --log-file %LOG% --output-dir models/base_ro_88m_pack_rewarm --repo-id flaviussteff/base-ro-88m-pack-rewarm
    )
    if errorlevel 1 goto :err
) else (
    echo        re-warm-pack exista, sar.
)

call :stamp "[6/9] SPP{MT}-pack: Base-pack@45k + 5.000 pasi cu reflectii (~6 h)"
if not exist models\spp_ro_88m_mt_pack\model.safetensors (
    if exist models\spp_ro_88m_mt_pack\checkpoints\checkpoint-* (
        py src/train_scale_pretrain.py %SPP_FLAGS% %MT_FLAGS% %MT_SPP_FLAGS% --resume --output-dir models/spp_ro_88m_mt_pack --repo-id flaviussteff/spp-ro-88m-mt-pack
    ) else (
        py src/train_scale_pretrain.py %SPP_FLAGS% %MT_FLAGS% %MT_SPP_FLAGS% --output-dir models/spp_ro_88m_mt_pack --repo-id flaviussteff/spp-ro-88m-mt-pack
    )
    if errorlevel 1 goto :err
) else (
    echo        MT-pack exista, sar.
)

call :stamp "[7/9] Filtered-pack - 50.000 pasi de la zero, loss mascat pe documentele semnalate (~60 h)"
if "%SKIPFILT%"=="1" (
    echo        nofiltered: sar.
    goto :states
)
if not exist models\filtered_ro_88m_pack\model.safetensors (
    if exist models\filtered_ro_88m_pack\checkpoints\checkpoint-* (
        py src/train_scale_pretrain.py %FILT_FLAGS% --steps 50000 --resume --output-dir models/filtered_ro_88m_pack --repo-id flaviussteff/filtered-ro-88m-pack
    ) else (
        py src/train_scale_pretrain.py %FILT_FLAGS% --steps 50000 --from-scratch --output-dir models/filtered_ro_88m_pack --repo-id flaviussteff/filtered-ro-88m-pack
    )
    if errorlevel 1 goto :err
) else (
    echo        Filtered-pack exista, sar.
)

:states
call :stamp "[8/9] Arhivez trainer_state.json"
if not exist evals\training_states mkdir evals\training_states
for %%M in (base_ro_88m_pack spp_ro_88m_pack base_ro_88m_pack_rewarm spp_ro_88m_mt_pack filtered_ro_88m_pack) do (
    if exist models\%%M\checkpoints\checkpoint-50000\trainer_state.json copy /Y models\%%M\checkpoints\checkpoint-50000\trainer_state.json evals\training_states\trainer_state_%%M.json >nul
    if exist models\%%M\checkpoints\checkpoint-5000\trainer_state.json copy /Y models\%%M\checkpoints\checkpoint-5000\trainer_state.json evals\training_states\trainer_state_%%M.json >nul
)

:evals
call :stamp "[9/9] Evaluari pentru toate modelele prezente (~1-2 h): ConstitutionEval-Ro v3 (4 protocoale), PPL held-out, cloze, prefixe adversariale"
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
