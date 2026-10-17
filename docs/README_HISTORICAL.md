# Multilingual next-word prediction: Telugu phase

Four models have completed five-epoch training and held-out evaluation: LSTM, Bi-LSTM, word CNN and gated word-character CNN. The Streamlit demo uses LSTM with hidden model and training controls.

Start with `./run_app.ps1`. See [FINAL_DELIVERY.md](FINAL_DELIVERY.md) for demo instructions and [FINAL_MODEL_RESULTS.md](FINAL_MODEL_RESULTS.md) for verified results. The current datasets are the native Telugu fields of the Telugu-LLM-Labs Alpaca and GPTeacher datasets, with provenance in `data_telugu_llm/provenance.json`.

The material below describes earlier experiments and is historical, not the current demo.

---
# Indic Federated Word and Sentence Prediction

## New Telugu training source

The new Telugu next-word training run uses the public instruction datasets
used by Telugu-Llama2: `Telugu-LLM-Labs/telugu_alpaca_yahma_cleaned_filtered_romanized`
and `Telugu-LLM-Labs/telugu_teknium_GPTeacher_general_instruct_filtered_romanized`.
It extracts `telugu_instruction`, `telugu_input`, and `telugu_output` (native script).
This trains the project's LSTM; it does not fine-tune the 7B Llama2 model or use TeSent.

`src/prepare_telugu_llm_data.py` verifies the downloaded CSV checksums, groups rows
by normalized instruction into deterministic 90/5/5 train/validation/test splits,
removes exact sentence overlaps, and builds the vocabulary only from training data.
The test split stays unused during training. Sources, licenses, counts, and hashes
are recorded in `data_telugu_llm/provenance.json`.

Run `run_telugu_llm_train.ps1 -PythonPath <python.exe> -Background` after preparation.
The new checkpoint is `outputs/telugu_llm_lstm.pt`, with progress in
`outputs/telugu_llm_train.log`. The frontend's Telugu mode uses this checkpoint
and the new corpus n-grams. It reloads a newly saved checkpoint on the next
prediction. Once all epochs finish, `finish_telugu_llm_train.ps1` evaluates the
held-out test split and saves `outputs/telugu_llm_test_metrics.json`. Older
training figures below do not describe this run.

Faculty-ready Telugu, Hindi, and English next-word plus sentence-continuation prediction system using PyTorch, LSTM, Flower federated learning, AI4Bharat IndicCorpV2, WikiText, and a simple Streamlit frontend.

## Current Improved Version

- Telugu, Hindi, and English cleaning, Unicode normalization, and word tokenization.
- AI4Bharat IndicCorpV2 Telugu (`tel_Telu`) and Hindi (`hin_Deva`) corpus support.
- WikiText English (`Salesforce/wikitext`, `wikitext-2-raw-v1`) corpus support.
- Large sharded preprocessing for big text files.
- Centralized LSTM baseline checkpoints for Telugu, Hindi, and English.
- Flower FedAvg server and client scripts.
- Agentic prediction router for cleaner Indic word outputs.
- Full sentence continuation suggestions for all three languages.
- Simple frontend with language selection, one input box, and readable outputs.
- One-command tri-lingual project self-check for demos.

## Current Trained Artifacts

```text
Telugu cleaned sentences: 50,648
Telugu training sequences: 1,718,124
Telugu vocabulary size: 30,000
Telugu LSTM epoch: 5
Telugu best validation loss: 6.8388

Hindi cleaned sentences: 141,367
Hindi training sequences: 2,691,266
Hindi vocabulary size: 30,000
Hindi LSTM epoch: 6
Hindi best validation loss: 7.1259

English cleaned sentences: 22,251
English training sequences: 1,680,444
English vocabulary size: 30,000
English LSTM epoch: 3
English best validation loss: 6.6101

Sequence length: 8
```

## Tested Environment

```text
torch: 2.13.0+cpu
streamlit: 1.59.1
flwr: 1.32.1
datasets: 5.0.0
numpy: 2.5.1
tqdm: 4.68.4
```

`pip check` passes with no broken requirements.

## Run Self Check

Use this before demo, report submission, or faculty review:

```powershell
cd C:\Users\sweth\Downloads\UROP
.\run_self_check.ps1 -RequireHealth
```

If the Streamlit app is not running, use:

```powershell
.\run_self_check.ps1
```

Latest full self-check result:

```text
Overall: PASS
Streamlit health: live on http://localhost:8501
Telugu sequences: 1,718,124
Hindi sequences: 2,691,266
English sequences: 1,680,444
Telugu checkpoint: epoch 5, vocab 30,000, best_val_loss 6.8388
Hindi checkpoint: epoch 6, vocab 30,000, best_val_loss 7.1259
English checkpoint: epoch 3, vocab 30,000, best_val_loss 6.6101
```

## Run Frontend

```powershell
cd C:\Users\sweth\Downloads\UROP
.\run_app.ps1 -Port 8501
```

Open:

```text
http://localhost:8501
```

The frontend is intentionally simple: language dropdown, one input box, `Predict`, 5 next-word outputs, and 5 sentence suggestions.

## Create Final Zip

To create a shareable package without `.venv`, `.git`, or cache files:

```powershell
.\create_release.ps1
```

This creates:

```text
Indic_FL_Word_Sentence_Final.zip
```

The zip includes source code, Telugu/Hindi/English data artifacts, trained LSTM checkpoints, report, run scripts, and final delivery notes.

## Run CLI Prediction

```powershell
cd C:\Users\sweth\Downloads\UROP
.\run_telugu_predict.ps1 -Text "నేను తెలుగు" -TopK 5
.\run_hindi_predict.ps1 -Text "मैं हिंदी" -TopK 5
.\run_english_predict.ps1 -Text "I am learning English" -TopK 5 -Sentences
```

Direct Python command:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\predict.py --language telugu --text "సర్వర్ ప్రతి రౌండ్ తర్వాత కొత్త మోడల్" --top-k 5 --sentences --verbose
.\.venv\Scripts\python.exe -X utf8 src\predict.py --language hindi --text "फेडरेटेड लर्निंग में" --top-k 5 --sentences --verbose
.\.venv\Scripts\python.exe -X utf8 src\predict.py --language english --text "federated learning protects" --top-k 5 --sentences --verbose
```

Example output:

```text
ఇచ్చిన వాక్యం: సర్వర్ ప్రతి రౌండ్ తర్వాత కొత్త మోడల్
తదుపరి పదాలు:
1. పంపిస్తుంది
2. పారామీటర్లను
3. సమీకరిస్తుంది
4. మెరుగుపడుతుంది
5. నవీకరిస్తుంది
వాక్య సూచనలు:
1. సర్వర్ ప్రతి రౌండ్ తర్వాత కొత్త మోడల్ పారామీటర్లను సమీకరిస్తుంది.
2. సర్వర్ ప్రతి రౌండ్ తర్వాత కొత్త మోడల్ నవీకరిస్తుంది.
3. సర్వర్ ప్రతి రౌండ్ తర్వాత కొత్త మోడల్ క్లయింట్ల నుండి వచ్చిన అప్డేట్లను కలుపుతుంది.
4. సర్వర్ ప్రతి రౌండ్ తర్వాత కొత్త మోడల్ గ్లోబల్ పనితీరును మెరుగుపరుస్తుంది.
5. సర్వర్ ప్రతి రౌండ్ తర్వాత కొత్త మోడల్ కొత్త పదాలను మరింత బాగా నేర్చుకుంటుంది.
```

## Important Files

```text
src/app.py                  Simple Streamlit frontend
src/agentic_predictor.py    Agentic Telugu/Hindi/English word and sentence predictor
src/predict.py              CLI prediction entry point
src/self_check.py           Tri-lingual project health and prediction checks
src/preprocess.py           Telugu/Hindi/English preprocessing
src/centralized_train.py    LSTM training
src/fl_server.py            Flower FedAvg server
src/fl_client.py            Flower client
outputs/centralized_lstm.pt Trained Telugu LSTM checkpoint
outputs/hindi_lstm.pt       Trained Hindi LSTM checkpoint
outputs/english_lstm.pt     Trained English LSTM checkpoint
```

## Rebuild Dataset

```powershell
.\.venv\Scripts\python.exe -X utf8 src\download_data.py --limit 50000 --output data\telugu_raw_large.txt
.\.venv\Scripts\python.exe -X utf8 src\build_training_corpus.py --input data\telugu_raw.txt --input data\telugu_raw_large.txt --repeat-first 25 --output data\telugu_training.txt
.\.venv\Scripts\python.exe -X utf8 src\preprocess.py --raw-path data\telugu_training.txt --max-vocab 30000 --seq-length 8 --shard-size 100000 --ngram-order 5

.\.venv\Scripts\python.exe -X utf8 src\download_data.py --language hindi --limit 50000 --output data_hindi\hindi_raw_large.txt
.\.venv\Scripts\python.exe -X utf8 src\build_training_corpus.py --language hindi --input data_hindi\hindi_raw.txt --input data_hindi\hindi_raw_large.txt --repeat-first 25 --output data_hindi\hindi_training.txt
.\.venv\Scripts\python.exe -X utf8 src\preprocess.py --language hindi --raw-path data_hindi\hindi_training.txt --max-vocab 30000 --seq-length 8 --shard-size 100000 --ngram-order 5

.\.venv\Scripts\python.exe -X utf8 src\download_data.py --language english --limit 50000 --output data_english\english_raw_large.txt
.\.venv\Scripts\python.exe -X utf8 src\build_training_corpus.py --language english --input data_english\english_raw.txt --input data_english\english_raw_large.txt --repeat-first 25 --output data_english\english_training.txt
.\.venv\Scripts\python.exe -X utf8 src\preprocess.py --language english --raw-path data_english\english_training.txt --max-vocab 30000 --seq-length 8 --shard-size 100000 --ngram-order 5
```

## Train Or Resume LSTM

Fresh training:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\centralized_train.py --epochs 5 --batch-size 512 --embedding-dim 64 --hidden-dim 128 --num-layers 1 --dropout 0.0 --validation-split 0.05 --output outputs\centralized_lstm.pt
.\.venv\Scripts\python.exe -X utf8 src\centralized_train.py --language hindi --resume outputs\hindi_lstm.pt --epochs 6 --max-samples 100000 --batch-size 256 --embedding-dim 96 --hidden-dim 192 --num-layers 2 --dropout 0.2 --save-every-batches 100 --output outputs\hindi_lstm.pt
.\.venv\Scripts\python.exe -X utf8 src\centralized_train.py --language english --resume outputs\english_lstm.pt --epochs 3 --max-samples 50000 --batch-size 256 --embedding-dim 96 --hidden-dim 192 --num-layers 2 --dropout 0.2 --save-every-batches 100 --output outputs\english_lstm.pt
```

Resume from the current checkpoint:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\centralized_train.py --resume outputs\centralized_lstm.pt --epochs 8 --batch-size 512 --embedding-dim 64 --hidden-dim 128 --num-layers 1 --dropout 0.0 --validation-split 0.05 --output outputs\centralized_lstm.pt
```

CPU training is slow on the full dataset. The current app is already usable for the demo because the agentic router combines project contexts, corpus n-grams, and trained LSTM checkpoints.

## Evaluate

```powershell
.\.venv\Scripts\python.exe -X utf8 src\evaluate.py --mode auto --top-k 5 --max-examples 1000 --raw-path data\telugu_training.txt
```

Latest quick validation sample:

```text
Examples: 100
Mode: auto
Top-1 accuracy: 0.930
Top-3 accuracy: 1.000
Top-5 accuracy: 1.000
```

## Federated Learning

Start server:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\fl_server.py --rounds 3 --min-clients 5
```

Start clients in separate terminals:

```powershell
.\.venv\Scripts\python.exe -X utf8 src\fl_client.py 0
.\.venv\Scripts\python.exe -X utf8 src\fl_client.py 1
.\.venv\Scripts\python.exe -X utf8 src\fl_client.py 2
.\.venv\Scripts\python.exe -X utf8 src\fl_client.py 3
.\.venv\Scripts\python.exe -X utf8 src\fl_client.py 4
```

## Report

See [PROJECT_REPORT.md](PROJECT_REPORT.md) for the academic report-style summary.
See [FINAL_DELIVERY.md](FINAL_DELIVERY.md) for quick demo and handoff notes.

