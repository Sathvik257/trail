# Multilingual Next-Word Prediction: Telugu phase

PyTorch next-word prediction with a Streamlit interface. Telugu is the current verified language. Legacy multilingual and federated scripts are retained for reference, not used in the current demo.

## Latest verified results

All rows use the same 200,092 held-out test examples. Higher accuracy is better; lower perplexity is better.

| Model / experiment | Top-1 | Top-5 | Perplexity |
|---|---:|---:|---:|
| LSTM, 5 epochs | 18.03% | 33.57% | 199.02 |
| Bi-LSTM, 5 epochs | 18.82% | 35.00% | 178.60 |
| Word CNN | 17.67% | 33.53% | 193.76 |
| Fine-tuned word CNN, 10 epochs | 18.68% | 34.70% | 183.54 |
| Gated word-character CNN | 18.55% | 34.68% | 167.13 |
| Fine-tuned Bi-LSTM v2 | 19.42% | 35.56% | 176.04 |
| Regularized Bi-LSTM v3 | 19.35% | 35.72% | 169.48 |
| Hybrid v1 | 20.62% | **37.63%** | Not measured |
| Longer-context hybrid v2 | **20.84%** | 37.46% | Not measured |

These are measured experimental results, not 80–90% accuracy. Exact metrics and validation-selection records are in `outputs/`.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run src/telugu_app.py
```

The app intentionally uses the original LSTM checkpoint, with model and training controls hidden. Best checkpoints are included. Open the local URL printed by Streamlit. No training dataset is required for this demo.

## Data and reproduction

Native Telugu instruction/input/output fields come from:

- [Telugu-LLM-Labs Alpaca](https://huggingface.co/datasets/Telugu-LLM-Labs/telugu_alpaca_yahma_cleaned_filtered_romanized): 28,910 records; provenance records CC BY 4.0.
- [Telugu-LLM-Labs GPTeacher](https://huggingface.co/datasets/Telugu-LLM-Labs/telugu_teknium_GPTeacher_general_instruct_filtered_romanized): 43,614 records; provenance records MIT.

Training: 3,483,073 examples. Validation: 198,928. Test: 200,092. Vocabulary: 20,000 tokens, fitted on training only. Context: eight words. Splits group normalized instructions and remove exact sentence overlap. See `data_telugu_llm/provenance.json` for source hashes and attribution. Respect upstream terms.

Downloaded CSVs and generated sequence shards are excluded from Git. Obtain the source CSVs as `alpaca.csv` and `gpteacher.csv` in a fresh directory `data_rebuilt/source/`, matching the checksums in `src/prepare_telugu_llm_data.py`.

```powershell
python src/prepare_telugu_llm_data.py --data-dir data_rebuilt
python src/train_telugu_comparison.py --data-dir data_rebuilt --output-dir outputs/reproduced --epochs 5
```

Improvement scripts expect prepared data at `data_telugu_llm` and verify parent-checkpoint hashes. Rebuild in a separate checkout: the included provenance file intentionally prevents overwriting an existing prepared directory. Historical machine paths in experiment records describe the original run, not installation requirements.

## Repository guide

- [Full architecture and flowcharts](docs/ARCHITECTURE.md): all models, formulas, training and prediction flow.
- `src/`: preparation, training, evaluation, model and UI source.
- `outputs/`: best checkpoints, hybrid count models, histories, configurations and metrics. Resume-checkpoint duplicates and runtime logs are excluded.
- `presentations/`: slide versions with formulas and flowcharts. They predate the newest accuracy experiments; use the results table above for current figures.
- [Historical README](docs/README_HISTORICAL.md): archived earlier description, including legacy experiments.
- `FINAL_MODEL_RESULTS.md`: original four-model benchmark. Later results are in the accuracy/hybrid output folders.

Baseline training: Adam, learning rate 0.001, batch 512, seed 42, five epochs. Bi-LSTM v2 fine-tuned at 0.0003 through epoch 10; validation selected epoch 9. V3 used AdamW, learning rate 0.0001, weight decay 0.01 and training-only dropout 0.15 through epoch 14; validation selected epoch 13. Hybrid parameters are selected by validation top-1.

## Limits

The vocabulary is fixed; unseen target words count as misses. Several plausible next words may exist, while evaluation checks one recorded target. Results use one seed and repeated reporting on the same test set; a fresh untouched evaluation set is needed for independent confirmation. Exact duplicate checks do not exclude semantic duplication. Novelty, gate-specific benefit, and 80–90% accuracy are not established. Training can take hours on CPU and cannot continue through shutdown.

