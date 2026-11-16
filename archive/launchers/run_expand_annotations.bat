@echo off
chcp 65001 >nul
cd /d "%~dp0"
title SPP-Ro: extindere sidecar v3 (adnotare Qwen, doar API)
if not exist logs mkdir logs
set LOG=logs\expand_v3.log
set MODEL=qwen-3.8-27b
set OUT=data\sidecar\reflections_v3.parquet
set RUNNING=data\sidecar\expand_v3.running

REM ---------------------------------------------------------------------------
REM  Extinderea sidecar-ului la ~10 la suta din documente (articolul: 10 la suta documente, 0,55 la suta tokeni).
REM  Adnoteaza documentele din data\clean\spp_candidates_v3.parquet (prag clasificator coborat, doar documente
REM  care NU sunt deja in reflections_v2.parquet) si le scrie in data\sidecar\reflections_v3.parquet,
REM  care porneste ca o COPIE a lui reflections_v2.parquet (v2 ramane neatins, generatia 2 ramane reproductibila).
REM  Nu foloseste GPU-ul: ruleaza in paralel cu antrenarea Base-pack. Reluabil: continua de unde a ramas.
REM  Chei: .env, CEREBRAS_API_KEYS=csk-...,csk-...  (cele fara credit sunt retrase automat; cand toate sunt
REM  epuizate scriptul se opreste cu progresul salvat -> adauga chei si ruleaza din nou acest .bat).
REM  La final scrie data\sidecar\expand_v3.done, pe care run_v3_pipeline.bat il asteapta inainte de SPP-pack.
REM ---------------------------------------------------------------------------
if not exist data\clean\spp_candidates_v3.parquet (
    echo [ERROR] Lipseste data\clean\spp_candidates_v3.parquet - ruleaza run_v3_pipeline.bat ^(pasul 1^).
    pause & exit /b 1
)
if exist data\sidecar\expand_v3.done (
    echo Extinderea este deja completa ^(data\sidecar\expand_v3.done exista^).
    pause & exit /b 0
)
if exist %RUNNING% (
    echo [ATENTIE] %RUNNING% exista: fie adnotarea ruleaza deja in alta fereastra, fie a fost inchisa brutal.
    echo           Daca nu exista alta fereastra "SPP-Ro: extindere sidecar v3", sterge fisierul si ruleaza din nou.
    pause & exit /b 1
)
if not exist %OUT% (
    echo Pornesc reflections_v3.parquet ca o copie a lui reflections_v2.parquet ...
    copy /Y data\sidecar\reflections_v2.parquet %OUT% >nul
    if errorlevel 1 ( echo [ERROR] copierea a esuat & pause & exit /b 1 )
)
echo ok > %RUNNING%
echo ===== START %date% %time% >> %LOG%
py src/generate_reflections_v2.py --candidates data/clean/spp_candidates_v3.parquet --out %OUT% --batch 3 --model %MODEL%
if errorlevel 1 goto :err

REM complete = at most 2 la suta of the candidates are still unannotated (rejected outputs are not retried)
py -X utf8 -c "import pandas as pd,sys; c=set(pd.read_parquet('data/clean/spp_candidates_v3.parquet',columns=['doc_key'])['doc_key']); d=set(pd.read_parquet('data/sidecar/reflections_v3.parquet',columns=['doc_key'])['doc_key']); import json,os; sk=set(json.load(open('data/sidecar/reflections_v2_skip_keys.json',encoding='utf-8'))) if os.path.exists('data/sidecar/reflections_v2_skip_keys.json') else set(); r=len(c-d-sk); print(f'candidati v3: {len(c)} | neadnotati inca: {r} ({100*r/max(1,len(c)):.1f} la suta)'); sys.exit(0 if r<=0.02*len(c) else 3)"
if errorlevel 1 (
    del %RUNNING% >nul 2>&1
    echo.
    echo [INCOMPLET] Au ramas documente neadnotate ^(sau verificarea a esuat^) - probabil cheile s-au epuizat.
    echo             Adauga chei noi in .env si ruleaza din nou run_expand_annotations.bat ^(continua de unde a ramas^).
    echo             Daca vrei sa continui pipeline-ul cu ce exista:   echo ok ^> data\sidecar\expand_v3.done
    echo ===== INCOMPLETE %date% %time% >> %LOG%
    pause
    exit /b 3
)
echo ok > data\sidecar\expand_v3.done
del %RUNNING% >nul 2>&1
py -X utf8 -c "import pandas as pd; d=pd.read_parquet('data/sidecar/reflections_v3.parquet'); n=len(d); print(f'reflections_v3.parquet: {n} documente ({100*n/1500000:.1f} la suta din corpus), severitate>=3: {int((d.severity>=3).sum())}, clf_flag=1: {int((d.clf_flag==1).sum())}')"
echo ===== DONE %date% %time% >> %LOG%
echo.
echo Gata. run_v3_pipeline.bat va continua automat cu SPP-pack cand ajunge acolo.
pause
exit /b 0

:err
del %RUNNING% >nul 2>&1
echo [ERROR] %date% %time% adnotarea s-a oprit ^(cod 2 = toate cheile epuizate; altfel vezi consola de mai sus^).
echo         Chei epuizate: adauga chei noi in .env si ruleaza din nou run_expand_annotations.bat - continua de unde a ramas.
echo ===== ERROR %date% %time% >> %LOG%
pause
exit /b 1
