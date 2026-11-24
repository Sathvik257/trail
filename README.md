# Multilingual Next-Word Prediction

A PyTorch research project for **Telugu, Hindi and English**, with a Streamlit demo, reproducible training scripts and saved evaluation evidence. Given a short phrase, the app suggests five next words using a language-specific model.

**Updated: 24 November 2026.** All three language baselines and their Bi-LSTM continuations are complete. The demo uses each language's **LSTM baseline**; the stronger research hybrids are separate experiments. This project trains word-level models from scratch; it does not serve a pretrained LLM.

## Quick start

Python **3.11+** is required by the preparation scripts; the project checks were run with Python 3.12. Git includes saved checkpoints, so the demo needs no dataset download or training. The repository is sizeable because research checkpoints are retained.

```powershell
git clone https://github.com/Sathvik257/trail.git
cd trail
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run src/app.py
```

On Linux/macOS, activate with `source .venv/bin/activate`. Open the local URL printed by Streamlit, choose a language, enter a phrase and click **Predict next word**. If PowerShell activation is restricted, use `.\.venv\Scripts\python.exe` directly. Windows launchers accept an explicit `-PythonPath`.

```powershell
python src/predict.py --language english --text "To stay healthy"
python src/predict.py --language hindi --model bilstm --text "स्वस्थ रहने के लिए"
python src/predict.py --language telugu --text "ఆరోగ్యంగా ఉండటానికి"
```

`--checkpoint PATH` selects another compatible neural checkpoint. `src/telugu_app.py` remains a compatibility entry point. Adding `?model=hybrid` to the local app URL enables the saved **Telugu hybrid v2** (20.84% top-1), not the newer discounted experiment.

## Measured results

Accuracy measures exact agreement with one recorded next word. Top-5 checks whether that target appears among five suggestions. These are saved held-out measurements, not a new evaluation performed during repository cleanup.

| Language | Demo LSTM top-1 | Continued Bi-LSTM top-1 | Hybrid v2 top-1 / top-5 | Test examples |
|---|---:|---:|---:|---:|
| Telugu | 18.03% | 19.42% | 20.84% / 37.46% | 200,092 |
| Hindi | 29.41% | 30.89% | 32.79% / 56.43% | 284,823 |
| English | 25.65% | 27.23% | 29.85% / 50.63% | 308,612 |

A completed Telugu discounted-count follow-up measured **20.85% top-1 / 37.84% top-5**. Its top-1 gain over hybrid v2 is only 0.0075 percentage points and is not evidence of a statistically significant improvement. Hindi retained its parent hybrid; the English discounted follow-up has no completed test result and is excluded.

See [the full results and source files](docs/RESULTS.md), [architecture](docs/ARCHITECTURE.md) and [reproduction instructions](docs/REPRODUCIBILITY.md). Different languages use different corpora, so their percentages are not directly comparable.

## Project structure

```text
src/                       Application, inference, data and experiment code
tests/                     Regression tests and real-checkpoint UI checks
docs/                      Architecture, results, reproduction and source guide
outputs/                   Selected checkpoints, count models and measured JSON evidence
data_telugu_llm/            Telugu provenance only (prepared data stays local)
data_hindi_hf/              Hindi provenance only
data_english_hf/            English provenance only
.github/workflows/         Automated regression checks
requirements.txt           Demo and test dependencies
requirements-training.txt  Additional corpus preparation dependencies
run_*.ps1 / run_*.bat       Windows launchers
```

The [source guide](docs/DEVELOPMENT.md) explains the modules and checks. Training scripts retain stable filenames and output paths so existing experiment evidence remains traceable. Old reports, slide versions, duplicate screenshots and unused federated/heuristic demo code were removed from the current Git tree; earlier versions remain in Git history. Local submission packages and working slide files are ignored.

## Verification

```powershell
python tests/run_checks.py             # Models, resume behavior and actual multilingual UI
python tests/run_checks.py --unit-only # Synthetic-data regression tests, no saved models needed
```

Checks cover prefix/target separation, unknown-target scoring, gradients, causal CNN behavior, checkpoint round-trips, training resume, all three language demos, empty input and the Telugu hybrid route. UI checks use Streamlit AppTest. They do not establish real-world accuracy or rerun every full held-out evaluation.

## Data and limitations

Training uses native-script Telugu fields from the Telugu-LLM-Labs Alpaca/GPTeacher sources, Hindi Alpaca translations and English Alpaca Cleaned. [Provenance and data instructions](docs/REPRODUCIBILITY.md) record source revisions, checksums, split sizes and recorded license information. Raw datasets and prepared sequence shards are excluded from Git.

- Vocabularies are fitted on training only, capped at 20,000 tokens, with eight-word contexts. Unknown target words count as accuracy misses.
- Splits group normalized instructions and remove exact sentence overlap; semantic duplication is still possible.
- The corpora are synthetic/translated instruction data, not representative typing logs. English preprocessing lowercases and limits characters; Indic input is restricted to its selected script.
- Results use one seed and repeatedly reported test sets. Independent evaluation, multiple seeds and controlled ablations remain necessary for stronger claims.
- Model probabilities are not calibrated guarantees. Neither 75–90% accuracy nor general conversational ability is established.
- Some experiment orchestration scripts use Windows-specific locking. Core inference and baseline training support other platforms; see the reproduction guide.

To export only the committed project (without local datasets, caches or submission builds), run `python src/package_project.py` or `.\create_release.ps1`. Commit changes first; exports represent `HEAD`.
