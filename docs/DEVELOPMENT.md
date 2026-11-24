# Source guide and maintenance

## Application and core modules

| Files | Responsibility |
|---|---|
| `src/app.py`, `src/telugu_app.py` | Canonical entry point and multilingual Streamlit UI |
| `src/predict.py`, `src/telugu_neural_predictor.py`, `src/telugu_hybrid_predictor.py` | CLI, neural inference and optional Telugu hybrid |
| `src/model.py`, `src/wordchar_model.py` | LSTM, Bi-LSTM, causal CNN and gated word-character CNN |
| `src/text_utils.py`, `src/preprocess.py`, `src/sequence_data.py`, `src/wordchar_data.py` | Normalization, vocabulary, prefix/target examples and data loading |
| `src/ngram_predictor.py` | Count-model utility used by corpus preprocessing |
| `src/prepare_telugu_llm_data.py`, `src/complete_hindi_english.py` | Main source preparation paths |
| `src/download_data.py`, `src/build_training_corpus.py`, `src/import_tesent.py` | Additional corpus utilities; not the published benchmark sources |
| `src/train_telugu_comparison.py`, `src/train_language_comparison.py` | Baseline training, split audits, evaluation and checkpoint persistence |
| `src/complete_telugu_*.py` | Windows orchestration for the original baseline/CNN runs |
| `src/improve_*.py`, `src/continue_*.py`, `src/train_*hybrid*.py`, `src/refine_hindi_english_hybrid.py` | Separate research follow-ups; see reproduction guide |
| `src/evaluate_telugu_counts.py`, `src/evaluate_telugu_frequency.py`, `src/verify_telugu_training.py` | Component/frequency diagnostics and saved baseline verification |
| `src/audit_paper_inference.py` | Post-hoc research coverage/latency audit; requires prepared data |
| `src/package_project.py` | Export committed Git contents without local working artifacts |

Historical Telugu module names remain to preserve imports and experiment traceability. Model definitions and experimental protocols are not changed merely to rename files. Some research scripts execute work at module import; invoke them as scripts, not libraries.

## Checks

`python tests/run_checks.py` runs syntax checks, regression suites and actual-checkpoint UI flows. `--unit-only` runs the synthetic-data suites used in GitHub Actions. Training regression tests use temporary tiny corpora and do not alter published checkpoints. UI receipts are generated under `outputs/checks/` (ignored); the older Telugu-only check also updates its tracked receipt.

The tests exercise target isolation, OOV scoring, recurrent gradients, causal convolution, word-character encoding, save/reload, resume and actual multilingual prediction. They are not full-scale retraining, independent accuracy validation or a complete audit of every research variant.

## Repository hygiene

Keep current explanations in README plus architecture, results and reproduction documents. Add completed experiment evidence with its source and selection record; exclude partial runs from result claims. Preserve historical metric files as evidence even if their machine paths are old. Avoid committing runtime logs, latest-checkpoint duplicates, raw corpora, environment folders, slide builds or submission copies.

The November cleanup removed old reports, presentation revisions and unused heuristic/federated code from the current tree. Their original versions remain in Git history, and the cleanup retained a local ignored `.local-archive/` copy. Active trained models and research measurements were retained. No history rewrite is required.
