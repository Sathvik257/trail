# Current delivery: multilingual next-word prediction, Telugu phase

The verified interface supports Telugu next-word prediction with five suggestions. Its default model is the completed LSTM. Model selection and training/evaluation panels remain hidden as requested. CNN checkpoints are research comparison artifacts and are not integrated into this interface.

## Start the demo

Run `./run_app.ps1 -Port 8501` from the project folder. Open http://127.0.0.1:8501/ and enter a Telugu phrase such as `ఆరోగ్యంగా ఉండటానికి`. Click **Predict next word**.

## Completed experiment

LSTM, Bi-LSTM, word CNN and gated word-character CNN each completed five epochs, held-out evaluation and checkpoint verification. See [FINAL_MODEL_RESULTS.md](FINAL_MODEL_RESULTS.md) for measured results and limitations.

## Verification

Use the project's Python environment to run `src/verify_telugu_training.py`, `src/check_telugu_ui.py`, `src/check_telugu_comparison.py`, `src/check_telugu_cnn.py` and `src/check_wordchar.py`. The frequency diagnostic is `src/evaluate_telugu_frequency.py`.

## Scope and remaining research

The character model is an implemented project contribution. Research originality needs a literature review. A trained ungated-fusion ablation and multiple seeds remain necessary to isolate the gate's value and estimate run variability. Further languages are future scope. The older multilingual/federated delivery description is archived in FINAL_DELIVERY_LEGACY.md and does not describe this demo.
