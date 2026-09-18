# ==============================================================================
# Master Pipeline Runner: Romanian LLM from Scratch & SPP on Windows PowerShell
# Hardware: NVIDIA GeForce RTX 3060 (12GB VRAM)
# ==============================================================================

param (
    [string]$Stage = "all",
    [int]$Steps = 5000,
    [int]$WikiArticles = 100000,
    [int]$NewsArticles = 25000,
    [int]$NewsYears = 5,
    [int]$FinewebShards = 0,
    [float]$TargetGb = 0
)

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   Romanian LLM from Token Zero (EPFL-SPP Adaptation) Pipeline" -ForegroundColor Cyan
Write-Host "   Hardware Target: RTX 3060 (12GB VRAM) | OS: Windows PowerShell" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# Ensure output encoding is UTF-8 for Romanian diacritics
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Check Python environment
Write-Host "`n[0/6] Verifying GPU and Python environment..." -ForegroundColor Yellow
py -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 1. Download Corpus
if ($Stage -eq "all" -or $Stage -eq "download") {
    Write-Host "`n[1/6] Stage 1: Downloading Romanian Corpus (Wikipedia + 5-Year News + FineWeb2)..." -ForegroundColor Yellow
    py src/download_corpus.py --wiki-articles $WikiArticles --news-articles $NewsArticles --years $NewsYears --fineweb-shards $FinewebShards --target-gb $TargetGb
}

# 2. Clean & Canonicalize Diacritics
if ($Stage -eq "all" -or $Stage -eq "clean") {
    Write-Host "`n[2/6] Stage 2: Cleaning, Diacritics Canonicalization (ș, ț) and Deduplication..." -ForegroundColor Yellow
    py src/clean_corpus.py
}

# 3. Train Tokenizer
if ($Stage -eq "all" -or $Stage -eq "tokenizer") {
    Write-Host "`n[3/6] Stage 3: Training Custom 16,384 BPE Romanian Tokenizer..." -ForegroundColor Yellow
    py src/train_tokenizer.py
}

# 4. Generate SPP Reflections
if ($Stage -eq "all" -or $Stage -eq "spp-annotate") {
    Write-Host "`n[4/6] Stage 4: Generating SPP Constitutional Reflections (10% subset)..." -ForegroundColor Yellow
    py src/spp_annotator.py --max-reflections 5000 --backend auto
}

# 5. Pretraining from Token Zero
if ($Stage -eq "all" -or $Stage -eq "train-spp") {
    Write-Host "`n[5/6] Stage 5: Pretraining SPP-Ro (Token Zero + Attention Block + RoPE Aliasing)..." -ForegroundColor Yellow
    py src/train_pretrain.py --mode spp --max-steps $Steps
}

if ($Stage -eq "train-base") {
    Write-Host "`n[5/6] Stage 5: Pretraining Base-Ro (Vanilla Next-Token Baseline)..." -ForegroundColor Yellow
    py src/train_pretrain.py --mode base --max-steps $Steps
}

# 6. Bias Evaluation & Unmasking
if ($Stage -eq "all" -or $Stage -eq "eval") {
    Write-Host "`n[6/6] Stage 6: Running Romanian BERT Baselines and Unmasking Probes..." -ForegroundColor Yellow
    py src/eval_biases.py --eval-bert
}

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host " Pipeline execution step completed successfully!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
