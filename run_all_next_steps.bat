@echo off
chcp 65001 >nul
title Pipeline Executie Automata — Licenta LLM SPP-Ro (Pasi 6 - 11)
cls

echo ============================================================================
echo   PIPELINE EXECUTIE AUTOMATA DE LA TOKEN ZERO LA EVALUARI FINALE
echo   Lucrare de Licenta: Model Raising from Token Zero (SPP-Ro)
echo   Autor: Flavius Stefan ^| Facultatea de Matematica si Informatica
echo ============================================================================
echo.
echo Acest script va executa complet si autonom toti pasii ramasi:
echo   [1/6] Incarcarea modelului preantrenat SPP-Ro-125M pe Hugging Face
echo   [2/6] Antrenarea adaptorului LoRA post-hoc pe Base-Ro-125M (Pasul 6)
echo   [3/6] Evaluarea Triadei de Aliniere Curata pe 37 perechi (Pasul 7)
echo   [4/6] Evaluarea Taxei de Aliniere (Perplexitate pe Wikipedia si Stiri) (Pasul 8)
echo   [5/6] Probing pe Prefixe Adversative si Demascare Jailbreaks (Pasul 9)
echo   [6/6] Studiul de Transferabilitate Externa pe RoGPT-780M (Pasul 11)
echo.
echo Data si ora de pornire: %date% %time%
echo ============================================================================
echo.

:: -----------------------------------------------------------------------------
:: ETAPA 1: Incarcare SPP-Ro-125M pe Hugging Face Hub
:: -----------------------------------------------------------------------------
echo [ETAPA 1/6] Incarcare model SPP-Ro-125M pe Hugging Face Hub...
py src\upload_to_hf.py --model spp --repo-id flaviussteff/spp-ro-125m
if errorlevel 1 (
    echo [!] Avertisment: Upload-ul HF a intampinat o problema sau nu este configurat token-ul.
    echo     Se continua cu etapele locale de antrenare si evaluare...
)
echo.

:: -----------------------------------------------------------------------------
:: ETAPA 2: Antrenare LoRA Post-Hoc (Pasul 6)
:: -----------------------------------------------------------------------------
echo ============================================================================
echo [ETAPA 2/6] Antrenare Model Control Post-Hoc LoRA (Base-Ro-125M + LoRA)...
echo ============================================================================
py src\train_lora_alignment.py
if errorlevel 1 (
    echo [EROARE CRITICA] Antrenarea LoRA a esuat!
    goto error_handler
)
echo.

:: Incarcare LoRA pe Hugging Face Hub
echo Incarcare adaptor Base-Ro-LoRA pe Hugging Face Hub...
py src\upload_to_hf.py --model lora --repo-id flaviussteff/base-ro-125m-lora
echo.

:: -----------------------------------------------------------------------------
:: ETAPA 3: Evaluarea Triadei de Aliniere (Pasul 7)
:: -----------------------------------------------------------------------------
echo ============================================================================
echo [ETAPA 3/6] Evaluare Triada de Aliniere Curata (Base vs. LoRA vs. SPP)...
echo ============================================================================
py src\eval_biases.py --triad --lang both
if errorlevel 1 (
    echo [EROARE] Evaluarea biasului a esuat!
    goto error_handler
)
echo.

:: -----------------------------------------------------------------------------
:: ETAPA 4: Verificarea Taxei de Aliniere (Pasul 8)
:: -----------------------------------------------------------------------------
echo ============================================================================
echo [ETAPA 4/6] Evaluare Alignment Tax (Perplexitate Wikipedia si Stiri)...
echo ============================================================================
py src\eval_alignment_tax.py
if errorlevel 1 (
    echo [EROARE] Evaluarea taxei de aliniere a esuat!
    goto error_handler
)
echo.

:: -----------------------------------------------------------------------------
:: ETAPA 5: Demascare Prefixe Adversative / Jailbreaks (Pasul 9)
:: -----------------------------------------------------------------------------
echo ============================================================================
echo [ETAPA 5/6] Rulare Suita de Probing Adversativ (15 tulpini diagnostice)...
echo ============================================================================
py src\eval_jailbreaks.py
if errorlevel 1 (
    echo [EROARE] Evaluarea adversativa a esuat!
    goto error_handler
)
echo.

:: -----------------------------------------------------------------------------
:: ETAPA 6: Studiul de Transferabilitate Externa pe RoGPT-780M (Pasul 11)
:: -----------------------------------------------------------------------------
echo ============================================================================
echo [ETAPA 6/6] Studiu de Transferabilitate pe Modelul RoGPT-780M (Dumitrescu)...
echo ============================================================================
py src\eval_external_transfer.py
if errorlevel 1 (
    echo [!] Avertisment: Studiul pe RoGPT-780M a intampinat o problema (posibil memorie VRAM).
)
echo.

:: -----------------------------------------------------------------------------
:: FINALIZARE CU SUCCES
:: -----------------------------------------------------------------------------
echo ============================================================================
echo   TOATE ETAPELE AU FOST EXECUTATE CU SUCCES!
echo ============================================================================
echo.
echo Toate tabelele LaTeX pentru lucrarea de licenta sunt generate in folderul 'evals/':
echo   1. evals\thesis_3way_alignment_triad.tex           (Triada de Aliniere)
echo   2. evals\thesis_crosslingual_bias_benchmark.tex    (Disparitate Lingvistica)
echo   3. evals\thesis_alignment_tax_table.tex            (Taxa de Aliniere PPL)
echo   4. evals\thesis_adversarial_prefix_unmasking.tex   (Demascare Jailbreaks)
echo   5. evals\thesis_external_transfer_rogpt.tex        (Transfer RoGPT-780M)
echo.
echo Poti lansa oricand interfata grafica Trio Arena demonstrativa ruland:
echo   >>> py app.py
echo.
echo Ora finalizarii: %date% %time%
echo ============================================================================
pause
exit /b 0

:error_handler
echo.
echo ============================================================================
echo [STOP] Executia a fost oprita din cauza unei erori. Verifica mesajele de mai sus.
echo ============================================================================
pause
exit /b 1
