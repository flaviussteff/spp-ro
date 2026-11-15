@echo off
chcp 65001 >nul
cd /d "%~dp0"
title SPP-Ro: upload to Hugging Face
REM ---------------------------------------------------------------------------
REM  Publishes the five models and the safety classifier on the Hugging Face Hub, each with a model card generated
REM  from the evaluation reports in evals\ (perplexity, cloze, ConstitutionEval-Ro):
REM    flaviussteff/spp-ro                   SPP{T0}                 <- models\spp_ro
REM    flaviussteff/spp-ro-base              Vanilla control         <- models\base_ro
REM    flaviussteff/spp-ro-filtered          Filtered baseline       <- models\filtered_ro
REM    flaviussteff/spp-ro-mt                SPP{MT}                 <- models\spp_ro_mt
REM    flaviussteff/spp-ro-base-rewarm       control for MT          <- models\base_ro_rewarm
REM    flaviussteff/spp-ro-safety-classifier                         <- models\safety_clf_ro
REM  checkpoints\ and training_args.bin are not uploaded. Existing repos are updated in place.
REM
REM  Once: hf auth login   (token with Write permission, from https://huggingface.co/settings/tokens)
REM  Usage: run_upload_hf.bat                      all repos, public
REM         run_upload_hf.bat private              all repos, private
REM         run_upload_hf.bat dry                  only print the model cards, upload nothing
REM         run_upload_hf.bat only spp-ro spp-ro-base ...   only the given repos
REM ---------------------------------------------------------------------------
set ARGS=
if /I "%~1"=="private" set ARGS=--private
if /I "%~1"=="dry" set ARGS=--dry-run
if /I "%~1"=="only" goto :only
goto :run

:only
shift
set ARGS=--only
:collect
if "%~1"=="" goto :run
set ARGS=%ARGS% %~1
shift
goto :collect

:run
py -X utf8 src\upload_to_hf.py %ARGS%
if errorlevel 1 (
    echo.
    echo [ERROR] The upload failed. If the message is about authentication: hf auth login
    pause
    exit /b 1
)
echo.
pause
