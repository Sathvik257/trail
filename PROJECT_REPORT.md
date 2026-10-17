> Current Telugu LSTM/Bi-LSTM work is documented in [TELUGU_EXPERIMENT.md](TELUGU_EXPERIMENT.md). The multilingual results below are historical and must not be used as results for the new experiment.

# Federated Learning Based Telugu Next Word Prediction System

## 1. Abstract

This project implements a Telugu next-word prediction system using an LSTM language model and a federated-learning-ready training pipeline. Telugu text is cleaned, tokenized, converted into word sequences, and used to train a PyTorch LSTM baseline. Flower is used to support federated learning, where clients train locally and send only model parameters to the server. The final version also includes an agentic prediction router that improves Telugu output clarity by combining project-specific context hints, corpus n-gram predictions, and the trained LSTM model.

## 2. Introduction

Next-word prediction is important for keyboard applications, typing assistants, and regional-language NLP systems. Telugu has rich morphology and many spelling/context variations, so predictions must be understandable and script-safe. This project focuses first on Telugu and keeps the design extensible for English and Hindi later.

## 3. Problem Statement

Centralized next-word prediction systems usually require collecting raw user text on a server. This can create privacy concerns. The project solves this by preparing a Telugu model that can be trained centrally for baseline comparison and then extended to federated learning, where raw client text remains local.

## 4. Objectives

- Build a Telugu next-word prediction pipeline.
- Train an LSTM model on Telugu text.
- Support large text preprocessing with sharded sequences.
- Add Flower-based federated learning with FedAvg.
- Preserve privacy by avoiding raw-text sharing in FL.
- Provide a simple working Streamlit frontend.
- Improve Telugu readability using an agentic prediction router.

## 5. Existing System

Existing next-word systems often depend on centralized data collection or large cloud language models. These systems can be difficult to adapt for privacy-aware regional-language typing. They may also produce unclear outputs when trained on noisy or domain-heavy text.

## 6. Proposed System

The proposed system uses a Telugu preprocessing pipeline, an LSTM language model, Flower federated learning, and a simple frontend. The improved version adds an agentic router that chooses cleaner suggestions from multiple local sources:

```text
Telugu input
  -> project-context hints
  -> curated Telugu examples
  -> large Telugu corpus n-grams
  -> trained LSTM fallback
  -> filtered top-k Telugu words
```

## 7. Dataset Description

```text
Dataset: AI4Bharat IndicCorpV2
Language split: tel_Telu
Downloaded lines: 50,000
Cleaned sentences: 50,648
Training sequences: 1,718,124
Vocabulary size: 30,000
Sequence length: 8 words
```

The final training corpus combines curated Telugu project examples with the larger downloaded Telugu corpus.

## 8. Preprocessing

The preprocessing stage normalizes Telugu Unicode, removes non-Telugu noise, splits sentences, tokenizes words, builds a capped vocabulary, and creates input-target sequence pairs. Large sequence data is saved into shards so training can work with a large corpus without loading every sequence into memory at once.

## 9. LSTM Model Architecture

```text
Input Telugu word IDs
  -> Embedding layer
  -> LSTM layer
  -> Fully connected layer
  -> Top-k next-word predictions
```

The current checkpoint uses a vocabulary of 30,000 words and a sequence length of 8.

## 10. Federated Learning Methodology

The Flower implementation divides the Telugu sequence dataset among clients. Each client trains locally and sends only model parameters to the server. The server aggregates updates using FedAvg.

```text
Client 0
Client 1
Client 2  -> local training -> FedAvg server -> updated global model
Client 3
Client 4
```

This design supports privacy-aware training because raw text data is not transferred to the central server.

## 11. Implementation

Important files:

```text
src/app.py                  Simple Streamlit frontend
src/agentic_predictor.py    Agentic Telugu prediction router
src/predict.py              CLI prediction entry point
src/self_check.py           Health and prediction checks
src/preprocess.py           Telugu preprocessing
src/centralized_train.py    LSTM training
src/fl_server.py            Flower FedAvg server
src/fl_client.py            Flower client
outputs/centralized_lstm.pt Trained LSTM checkpoint
```

## 12. Results and Analysis

Current trained model:

```text
Saved epoch: 5
Training samples: 1,632,218
Validation samples: 85,906
Best validation loss: 6.8388
```

Example predictions from the improved version:

```text
Input: నేను తెలుగు
Output: భాషను, నేర్చుకుంటున్నాను, మాట్లాడతాను, చదువుతున్నాను, పదాలను

Input: సర్వర్ ప్రతి రౌండ్ తర్వాత కొత్త మోడల్
Output: పంపిస్తుంది, పారామీటర్లను, సమీకరిస్తుంది, మెరుగుపడుతుంది, నవీకరిస్తుంది

Input: మొబైల్ కీబోర్డ్ తెలుగు పదాలను
Output: తదుపరి, పదాన్ని, సూచిస్తుంది, చూపిస్తుంది, త్వరగా

Input: తెలంగాణలో కరోనా వైరస్ కారణంగా
Output: కేసులు, ప్రజలు, ఆంక్షలు, ఆసుపత్రులు, పరీక్షలు
```

The direct project dependencies were checked with `pip check`, and no broken requirements were found. The main packages are installed and working: PyTorch, Flower, datasets, Streamlit, NumPy, and tqdm.

Latest quick validation:

```text
Examples: 100
Mode: auto
Top-1 accuracy: 0.930
Top-3 accuracy: 1.000
Top-5 accuracy: 1.000
Self-check: PASS
Streamlit health: live on http://localhost:8501
```

## 13. Advantages

- Telugu-focused NLP pipeline.
- Handles large Telugu text with sequence shards.
- Supports centralized and federated learning.
- Protects raw text privacy during federated training.
- Includes CLI, frontend, and self-check tools.
- Produces clearer demo predictions through the agentic router.

## 14. Limitations

- Full LSTM training is slow on CPU.
- The large corpus is news-heavy, so pure neural predictions may lean toward news vocabulary.
- Telugu next-word prediction cannot be perfectly accurate for every open-ended sentence.
- The current frontend is intentionally simple for testing and demonstration.

## 15. Future Scope

- Train more epochs on GPU.
- Save and compare federated global checkpoints.
- Add Hindi and English modes.
- Add transformer-based Telugu models.
- Add formal test-set metrics.
- Package the app for deployment.

## 16. Conclusion

The project is now a complete Telugu next-word prediction system with preprocessing, centralized LSTM training, federated learning support, an agentic prediction router, CLI prediction, a simple Streamlit frontend, and a one-command self-check. It is suitable for academic presentation and can be extended into a privacy-aware multilingual keyboard prediction system.

