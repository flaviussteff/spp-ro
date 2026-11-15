# models/ — what each folder is

All models have 88,099,584 parameters (Llama-style, 12 layers, hidden 768, GQA 12/4, 1,024-token context, 16k vocabulary, tied embeddings). The paths are fixed in `src/config.py` (`ALL_MODELS`). The folder is git-ignored; I publish the models on Hugging Face with `py src/upload_to_hf.py`.

All five read the same document stream (seed 42), 64 windows × 1,024 positions per step, ≈3.28 billion tokens in 50,000 steps (≈66 h each on an RTX 3060).

| Folder | Reported name | Hugging Face | What it is | Reference |
|---|---|---|---|---|
| `spp_ro/` | **SPP-Ro** | `flaviussteff/spp-ro` | SPP{T0}: 50,000 steps from scratch; the 150,000 annotated documents (10 %) carry their reflection inline, EPFL head-cap rule | Base-Ro |
| `base_ro/` | **Base-Ro** | `flaviussteff/spp-ro-base` | Vanilla control: the same stream, plain documents; `checkpoints_keep/checkpoint-45000` is the starting point of MT and re-warm | — |
| `filtered_ro/` | **Filtered-Ro** | `flaviussteff/spp-ro-filtered` | Filtered baseline: the same stream, the 75,000 classifier-flagged documents stay in context but receive no loss | Base-Ro |
| `spp_ro_mt/` | **SPP-Ro-MT** | `flaviussteff/spp-ro-mt` | SPP{MT}: Base-Ro at step 45,000 + 5,000 steps on the annotated stream (67.6 % annotated sequences, loss on the reflection only), as in the paper's midtraining | Base-Ro-rewarm |
| `base_ro_rewarm/` | **Base-Ro-rewarm** | `flaviussteff/spp-ro-base-rewarm` | control for MT: Base-Ro at step 45,000 + 5,000 steps with the same fresh LR schedule (peak 5e-5), no reflections | — |
| `safety_clf_ro/` | — | `flaviussteff/spp-ro-safety-classifier` | the BERT-Ro safety classifier + `threshold.json` (AUC 0.93 on dev; threshold 0.5 for the first 103k documents, 0.1 for the expansion); defines the annotated set, the Filtered mask and the "flagged" slice of the perplexity evaluation | — |

Each model folder holds the final model (`model.safetensors`), the tokenizer, `README.md` (model card generated from the run) and, in `checkpoints/`, the last two checkpoints with `stream_state.json` (the exact position of the data stream, for resuming). The `trainer_state.json` of every training is copied to `evals/training_states/`.

My earlier experiments (models with synthetic reflections, the un-packed run stopped at step 12,500, the first null MT attempt) are in `archive/v1/models/` and `archive/v2/models/`; they are neither evaluated nor reported.
