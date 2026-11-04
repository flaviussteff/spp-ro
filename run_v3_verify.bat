@echo off
chcp 65001 >nul
cd /d "%~dp0"
title ConstitutionEval-Ro v3: validitate + scorare modele
if not exist logs mkdir logs
set LOG=logs\v3_verify_and_eval.log

REM ---------------------------------------------------------------------------
REM  run_v3_verify.bat   dupa ce benchmark-ul v3 e construit (data\eval\constitution_eval_ro_v3.jsonl):
REM    [1/2] tabelul de validitate pe cei 600 de itemi (Qwen 27B blind, gpt-oss-120B open-book, Qwen2.5-7B local;
REM          4 rotatii, vot majoritar) -> evals\constitution_eval_v3_validity.json   (~1 h, API Cerebras + GPU)
REM          Reluabil: modelele deja scrise in fisier sunt sarite.
REM    [2/2] scorarea tuturor modelelor din models\ pe benchmark (continuare + litere, pasiv + deliberativ)
REM          -> evals\thesis_constitution_eval_v3_report.json + tabele .tex                       (~20 min, GPU)
REM  Live: watch_log.bat logs\v3_verify_and_eval.log
REM ---------------------------------------------------------------------------

if not exist data\eval\constitution_eval_ro_v3.jsonl (
    echo [ERROR] Lipseste data\eval\constitution_eval_ro_v3.jsonl - ruleaza intai build_constitution_eval_v3.py
    pause
    exit /b 1
)

echo ============================================================================
echo   V3 VERIFY + EVAL  started %date% %time%   log: %LOG%
echo ============================================================================
echo ===== V3 VERIFY START %date% %time% >> %LOG%

echo [1/2] %time%  tabelul de validitate (~1 h) ...
py -X utf8 src\build_constitution_eval_v3.py --stage verify --workers 12 --workers-guard 4 >> %LOG% 2>&1
if errorlevel 1 (
    echo [WARN] verify a esuat - vezi %LOG%
    echo [WARN] verify failed >> %LOG%
) else (
    echo        %time%  OK verify -^> evals\constitution_eval_v3_validity.json
)

echo [2/2] %time%  scorarea modelelor pe v3 (~20 min) ...
py -X utf8 src\eval_constitution_v3.py >> %LOG% 2>&1
if errorlevel 1 (
    echo [WARN] eval_constitution_v3 a esuat - vezi %LOG%
    echo [WARN] eval failed >> %LOG%
) else (
    echo        %time%  OK eval -^> evals\thesis_constitution_eval_v3_report.json
)

echo.
echo ============================================================================
echo   DONE %date% %time%  - rezultate in evals\ , log %LOG%
echo ============================================================================
echo ===== V3 VERIFY DONE %date% %time% >> %LOG%
pause
exit /b 0
