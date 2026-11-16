@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title SPP-Ro - re-evaluare dupa corecturi
if not exist logs mkdir logs

REM ---------------------------------------------------------------------------
REM  Refacerea evaluarilor dupa corecturile din 16.11.2026 (NU reantreneaza nimic: modelele raman cele din models\).
REM
REM   run_fixes.bat           tot: auto-teste CPU -> backup -> set held-out v3 (doar daca nu exista deja)
REM                           -> perplexitate -> teste pereche -> cloze -> ConstitutionEval-Ro -> sonda adversariala
REM                           -> evals\RESULTS.md + sectiunile generate din README.md
REM   run_fixes.bat rebuild   reconstruieste setul held-out chiar daca exista deja versiunea 3
REM   run_fixes.bat report    doar ultimul pas (RESULTS.md + README), fara GPU
REM
REM  Durata: ~1-1,5 h pe RTX 3060 (ConstitutionEval-Ro e pasul cel mai lung). Tot ce scriu scripturile merge in
REM  logs\fixes.log; urmarire live din alta fereastra:
REM      powershell -NoProfile -Command "Get-Content logs\fixes.log -Encoding UTF8 -Wait -Tail 30"
REM  Nu publica nimic pe Hugging Face si nu face commit: le faci tu dupa ce verifici rezultatele.
REM ---------------------------------------------------------------------------
set LOG=logs\fixes.log
REM scripturile scriu diacritice: fara UTF-8, iesirea redirectata in log ar folosi cp1252 si ar putea esua
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set REBUILD=0
set REPORTONLY=0
for %%A in (%1 %2) do (
    if /I "%%A"=="rebuild" set REBUILD=1
    if /I "%%A"=="report" set REPORTONLY=1
)
for /f "usebackq delims=" %%i in (`py -c "import datetime;print(datetime.datetime.now().strftime('%%Y%%m%%d_%%H%%M'))"`) do set STAMP=%%i
if not defined STAMP set STAMP=backup
set BACKUP=data\eval\backup_before_fix_%STAMP%

call :banner "SPP-Ro: re-evaluare dupa corecturi   rebuild=%REBUILD% report=%REPORTONLY%   %date% %time%"
echo ===== FIXES START %date% %time% args=%1 %2 >> %LOG%
if "%REPORTONLY%"=="1" goto :report

call :stamp "[0/8] Auto-teste CPU: packing, pozitii, statistici (test_packing) + reluarea din checkpoint (test_resume)"
py src\test_packing.py >> %LOG% 2>&1
if errorlevel 1 (
    echo [ERROR] test_packing.py a esuat.
    goto :err
)
echo        OK test_packing.py
py src\test_resume.py >> %LOG% 2>&1
if errorlevel 1 (
    echo [ERROR] test_resume.py a esuat.
    goto :err
)
echo        OK test_resume.py

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > logs\gpu_mem.tmp 2>nul
set GPUMEM=
set /p GPUMEM=<logs\gpu_mem.tmp
if defined GPUMEM if !GPUMEM! GTR 3000 (
    echo [ATENTIE] GPU-ul are deja !GPUMEM! MiB ocupati - probabil ruleaza un antrenament.
    echo           Opreste-l intai. Apasa o tasta pentru a continua oricum sau inchide fereastra.
    pause >nul
)

call :stamp "[1/8] Backup al setului held-out vechi si al rapoartelor in %BACKUP%"
if not exist "%BACKUP%" mkdir "%BACKUP%"
if not exist "%BACKUP%\evals" mkdir "%BACKUP%\evals"
for %%F in (heldout_fineweb.parquet heldout_stats.json heldout_clf_scores.parquet) do (
    if exist data\eval\%%F copy /Y data\eval\%%F "%BACKUP%\%%F" >nul
)
copy /Y evals\*.json "%BACKUP%\evals\" >nul 2>&1
copy /Y evals\*.tex "%BACKUP%\evals\" >nul 2>&1
if exist evals\RESULTS.md copy /Y evals\RESULTS.md "%BACKUP%\evals\" >nul
copy /Y README.md "%BACKUP%\README.md" >nul
echo        copiate in %BACKUP% (setul vechi ramane pe loc: build_heldout_set.py il reverifica inainte sa-l inlocuiasca)

call :stamp "[2/8] Setul held-out v3: normalizat exact ca si corpusul, verificat cu cheia de deduplicare a corpusului (CPU, ~5-10 min)"
set NEED=1
if "%REBUILD%"=="0" (
    py -c "import json,sys; sys.exit(0 if json.load(open('data/eval/heldout_stats.json',encoding='utf-8')).get('version',0)>=3 else 1)" >nul 2>&1
    if not errorlevel 1 set NEED=0
)
if "!NEED!"=="1" (
    call :run "src\build_heldout_set.py" || goto :err
) else (
    echo        heldout_stats.json are deja versiunea 3 - sar ^(run_fixes.bat rebuild il reface^)
)

call :stamp "[3/8] Perplexitate pe tot setul held-out + feliile semnalat/nesemnalat + pierderi per document (GPU, ~15-25 min)"
call :run "src\eval_alignment_tax.py" || goto :err

call :stamp "[4/8] Teste pereche pe documente: bootstrap, testul semnului, Holm (CPU, secunde)"
call :run "src\eval_alignment_paired.py" || goto :err

call :stamp "[5/8] Cloze held-out, 1.000 itemi, McNemar fata de referinta (GPU, ~5 min)"
call :run "src\eval_capabilities.py --n 1000" || goto :err

call :stamp "[6/8] ConstitutionEval-Ro: 4 protocoale, prompt deliberativ corectat, McNemar / sign-flip exact + Holm (GPU, ~40-60 min)"
call :run "src\eval_constitution.py" || goto :err

call :stamp "[7/8] Sonda cu prefixe adversariale (GPU, cateva minute; cifrele nu se schimba, se reimprospateaza descrierile)"
call :run "src\eval_jailbreaks.py" || goto :err

:report
call :stamp "[8/8] evals\RESULTS.md + sectiunile generate din README.md (Summary, Results)"
echo ----- %date% %time% src\report_results.py >> %LOG%
py src\report_results.py > logs\report_results.tmp 2>&1
set RC=!errorlevel!
type logs\report_results.tmp >> %LOG%
type logs\report_results.tmp
if not "!RC!"=="0" (
    echo   [ERROR] src\report_results.py a esuat - vezi %LOG%
    goto :err
)

call :banner "GATA %date% %time%"
echo   Rezultate:   evals\RESULTS.md  ^(tabelele LaTeX: evals\*.tex^)
echo   README:      git diff README.md   ^(doar blocurile dintre markerii SUMMARY / RESULTS sunt regenerate^)
if exist "%BACKUP%" echo   Backup:      %BACKUP%
echo   Log:         %LOG%
echo   Liniile [warn] de mai sus ^(daca exista^) spun ce raport e vechi si ce trebuie rerulat.
echo.
echo   Pasii urmatori, manual:
echo     1. citeste evals\RESULTS.md si verifica avertismentele de mai sus ^(daca exista^)
echo     2. py src\upload_to_hf.py --dry-run      ^(model cards generate din noile rapoarte^)
echo     3. py src\upload_to_hf.py                ^(publicare; cere hf auth login^)
echo     4. git add -A ^&^& git commit             ^(cand esti multumit^)
echo ===== FIXES DONE %date% %time% >> %LOG%
pause
exit /b 0

:run
echo   %time%  ruleaza %~1 ...
echo ----- %date% %time% %~1 >> %LOG%
py %~1 >> %LOG% 2>&1
if errorlevel 1 (
    echo   [ERROR] %~1 a esuat - vezi %LOG%
    echo [ERROR] %~1 failed >> %LOG%
    exit /b 1
)
echo   %time%  OK %~1
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
echo [ERROR] %date% %time% Un pas a esuat; nimic din ce urmeaza nu a rulat. Ultimele linii din %LOG%:
echo ===== ERROR %date% %time% >> %LOG%
powershell -NoProfile -Command "Get-Content '%LOG%' -Encoding UTF8 -Tail 40"
echo.
if exist "%BACKUP%" echo Fisierele vechi sunt in %BACKUP%.
echo Dupa ce rezolvi cauza, ruleaza din nou run_fixes.bat.
pause
exit /b 1
