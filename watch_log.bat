@echo off
chcp 65001 >nul
cd /d "%~dp0"
REM Live view of a running pipeline log. Usage: watch_log.bat [logs\generate_v2.log | logs\v2_pipeline.log]
set F=%1
if "%F%"=="" set F=logs\generate_v2.log
title watching %F%
echo Urmaresc %F%  (Ctrl+C pentru a iesi; antrenarea NU se opreste)
echo.
powershell -NoProfile -Command "Get-Content -Path '%F%' -Tail 40 -Wait | ForEach-Object { $_ -replace '\r','' }"
