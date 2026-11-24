# Data and reproduction

## Environment

Use Python 3.11 or later (the preparation code uses `hashlib.file_digest`); validation used Python 3.12. Install `requirements.txt` for inference/tests and `requirements-training.txt` for preparation. Baseline trainers are CPU-oriented. Some continuation/hybrid orchestrators import `msvcrt` or Windows power-management APIs and therefore require Windows. Training can take hours and cannot continue while the machine is shut down.

## Sources and splits

The links below identify the datasets; license labels are recorded provenance, not a new legal review.

| Language | Source | Train / validation / test sequences | Provenance |
|---|---|---|---|
| Telugu | [Alpaca](https://huggingface.co/datasets/Telugu-LLM-Labs/telugu_alpaca_yahma_cleaned_filtered_romanized) + [GPTeacher](https://huggingface.co/datasets/Telugu-LLM-Labs/telugu_teknium_GPTeacher_general_instruct_filtered_romanized) | 3,483,073 / 198,928 / 200,092 | [Hashes and attribution](../data_telugu_llm/provenance.json) |
| Hindi | [Hindi Alpaca](https://huggingface.co/datasets/iamshnoo/alpaca-cleaned-hindi) | 4,998,180 / 280,957 / 284,823 | [Pinned revision and hash](../data_hindi_hf/provenance.json) |
| English | [Alpaca Cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) | 5,159,864 / 285,386 / 308,612 | [Pinned revision and hash](../data_english_hf/provenance.json) |

Recorded labels: Telugu Alpaca and English Alpaca CC BY 4.0; Telugu GPTeacher MIT; Hindi license unspecified in the recorded card. Verify applicable upstream terms before redistribution. The repository excludes raw source records and prepared corpora.

Instruction hashes assign deterministic 90/5/5 groups. Exact sentence intersections are removed before a training-only vocabulary is fitted. Actual sequence ratios differ from instruction ratios. Hindi/English preparation also rejects sentences containing a four-token phrase repeated at least four times.

## Telugu baseline reproduction

Download the Telugu source CSVs into a **fresh** `data_rebuilt/source/` directory as `alpaca.csv` and `gpteacher.csv`. The preparation script verifies the exact SHA-256 values listed in its `SOURCES` constant; different downloads fail rather than silently changing the experiment.

```powershell
python -m pip install -r requirements-training.txt
python src/prepare_telugu_llm_data.py --data-dir data_rebuilt
python src/train_telugu_comparison.py --data-dir data_rebuilt --output-dir outputs/reproduced --epochs 5
```

Baseline configuration: seed 42, eight-word context, vocabulary cap 20,000, embeddings 64, recurrent hidden size 128, batch 512, Adam at 0.001, five epochs. Select checkpoints by validation cross-entropy. For isolated CNN experiments, add `--cnn-only` or `--wordchar-only` and choose another output directory.

The checked-in `data_telugu_llm/provenance.json` intentionally prevents preparing over that directory. To reproduce continuation scripts that assume the canonical data path, work in a separate checkout, move its provenance record aside, place verified source CSVs in `data_telugu_llm/source/`, then prepare that directory. Never overwrite the only copy of existing prepared data.

## Hindi and English baseline reproduction

On Windows:

```powershell
python src/complete_hindi_english.py
```

This downloads the pinned source revisions, prepares both languages, then runs each five-epoch LSTM/Bi-LSTM comparison sequentially. Provenance-only directories are not considered prepared data. Existing compatible latest checkpoints are resumed. To run a baseline in isolation after preparation:

```powershell
python src/train_language_comparison.py --data-dir data_hindi_hf --output-dir outputs/hindi_reproduced --epochs 5
```

Downloaded source hashes should match the committed provenance before comparing results. Use fresh output directories to avoid mixing experiments. Resume rejects changed training configuration/data, including recorded PyTorch version.

## Follow-up experiments

Run these only when you intend a potentially lengthy training/evaluation job. They use fixed canonical directories and validate prepared-data hashes against parent artifacts.

| Experiment | Entry point | Prerequisite |
|---|---|---|
| Telugu Bi-LSTM continuation | `src/improve_telugu_accuracy.py` | Telugu baseline and prepared data |
| Telugu regularized continuation | `src/improve_telugu_regularized.py` | Prior Telugu continuation |
| Word CNN improvement / continuation | `src/improve_telugu_cnn_accuracy.py`, `src/continue_telugu_cnn_accuracy.py` | Corresponding parent CNN |
| Word-character improvement | `src/improve_telugu_wordchar_accuracy.py` | Parent word-character model |
| Telugu hybrids | `src/train_telugu_hybrid.py`, `src/train_telugu_hybrid_long.py` | Telugu continued Bi-LSTM |
| Hindi/English continuation | `src/continue_hindi_english.py` | Both baselines |
| Hindi/English hybrid / refinement | `src/train_hindi_english_hybrid.py`, `src/refine_hindi_english_hybrid.py` | Both continued Bi-LSTMs / fitted counts |
| Discounted-count follow-up | `src/train_discounted_hybrid.py` | Continued models and parent counts for all languages |
| Telugu count-only diagnostic | `src/evaluate_telugu_counts.py` | Telugu v2 counts and prepared data |

Checkpoints retain optimizer state for continuation. Best checkpoints and fitted counts are versioned; duplicate latest/resume checkpoints, raw data, logs and locks remain local. Load only trusted pickle/checkpoint artifacts. A historical `status.json` is a run receipt, not proof that a process is currently running.
