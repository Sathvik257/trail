# Telugu next-word prediction: LSTM and Bi-LSTM

Project scope: multilingual next-word prediction. Current implementation phase: Telugu. Hindi, English and improved architectures are planned for later phases. The prediction interface uses actual neural model outputs; no hand-written completions or n-gram lookup is credited as neural accuracy.

## Dataset and experiment

The existing supplied project corpus is `data_telugu_llm`: native-script Telugu instruction, input and output fields from the Telugu-LLM-Labs Alpaca and GPTeacher datasets. This is next-word training from their text, not Llama model fine-tuning. Source details and licenses are recorded in `data_telugu_llm/provenance.json`.

| Split | Sentences | Next-word examples |
|---|---:|---:|
| Train | 312,456 | 3,483,073 |
| Validation | 18,294 | 198,928 |
| Test | 18,662 | 200,092 |

The existing preparation groups normalized instructions into deterministic 90/5/5 splits and removes exact sentence overlap. The comparison runner independently checks exact sentence disjointness and fingerprints the text, vocabulary and sequence shards. This does not rule out semantic paraphrase overlap. Vocabulary: 20,000 entries fitted to training data. Context: up to 8 preceding words, left padded. Both models process identical prefix/target pairs, with the same shuffled order, seed, batch size and optimizer.

- LSTM: 64-dimensional embeddings, one recurrent layer, 128 hidden units.
- Bi-LSTM: 64-dimensional embeddings, one layer, 128 hidden units in each direction; concatenate both terminal hidden states.
- Adam, learning rate 0.001, gradient clipping at 1.0, seed 42, batch size 512.
- The initial run requests one full epoch per model on all training examples. It is a baseline, not evidence of convergence. More epochs can be requested explicitly with the same runner.
- The backward direction sees only the already typed prefix in reverse order. The target/future sentence is never part of the input.
- Bi-LSTM has more parameters with these settings. This is an equal-data/equal-epoch comparison, not a parameter-matched or compute-matched experiment.

Checkpoints are saved periodically and after epochs. Resume restores optimizer and batch position with deterministic shuffling. The best validation-loss checkpoint is used for test evaluation and the interface. Test data never drives gradient updates or checkpoint selection. Repeatedly extending training after inspecting test scores would turn that test into a development set; use validation alone to decide future training budgets.

## Results and how to run

Do not quote old README/report accuracy values for this experiment. Authoritative results are generated only after completed training:

- `outputs/telugu_comparison/status.json`: live state; completion requires `status: complete`.
- `lstm_test_metrics.json` and `bilstm_test_metrics.json`: measured test metrics.
- `comparison.json`: combined results, written after both models finish.
- `lstm_best.pt` and `bilstm_best.pt`: validation-selected inference checkpoints.
- `experiment.json`: hyperparameters, split hashes and environment.

```powershell
.\run_telugu_comparison.ps1 -Epochs 1
.\run_app.ps1
.\run_telugu_predict.ps1 -Model lstm -Text "ఆరోగ్యంగా ఉండటానికి"
.\run_telugu_predict.ps1 -Model bilstm -Text "ఆరోగ్యంగా ఉండటానికి"
```

Do not start a second runner in the same output directory while one is training. After it exits, `-Epochs 3` resumes until three total epochs, not three additional epochs. Use Python's `--output-dir` argument for independent experiments. The scripts use the local `.venv`, or the existing Downloads/UROP environment if the project was copied. `-PythonPath` overrides this selection.

Metrics include top-1, top-3 and top-5 actual-word accuracy, target out-of-vocabulary rate, known-target accuracy and vocabulary perplexity. Special tokens are excluded from word rankings; unknown targets count as misses. Perplexity still uses the fixed vocabulary with unknown words collapsed into `<unk>`, so it is not an open-vocabulary language-model score. Inference shows original softmax probabilities rather than renormalizing away unknown-word probability.

Run regression checks with your Python interpreter and `src/check_telugu_comparison.py`. They exercise target exclusion, both Bi-LSTM terminal states and gradients, unknown-word scoring, completed training, checkpoint loading and resuming both architectures.

## What is the novelty?

**The current contribution is an implemented Telugu-focused prediction system and a reproducible comparison, not a new LSTM algorithm.** A defensible project description is:

> We implement and evaluate Telugu next-word prediction using prefix-only LSTM and Bi-LSTM encoders on a shared native-script corpus, with instruction-grouped held-out evaluation, explicit vocabulary-coverage reporting and a model-selectable interface.

The useful engineering contributions are the consistent Telugu preprocessing, a controlled comparison with no future-word access, actual-word evaluation that does not credit unknown-token guesses, resumable training, and a working interface. These are project contributions, not evidence of a research-first method or superiority over published systems.

Prior work already includes Telugu next-word prediction with LSTM: [Next Word Prediction in Telugu Sentences Using Recurrent Neural Networks](https://www.tojqi.net/journal/article/download/3587/2446/3924). The authors' [Autonext repository](https://github.com/HAIx-Lab/Autonext) also implements LSTM/BiLSTM/GRU and statistical predictive-text methods for several Indic languages. Therefore, neither choosing Telugu nor comparing LSTM and Bi-LSTM alone establishes algorithmic novelty.

For a stronger future research contribution, test a **Telugu character-aware word representation** that combines word embeddings with a character encoder, motivated by inflection and rare forms. Compare word-only versus character-aware LSTM/Bi-LSTM, stratify results by rare words and unknown context words, and measure model size and typing latency. A character-aware input alone does not let a fixed word-output vocabulary generate unseen words; that requires a subword or character output mechanism. This extension is proposed, not implemented or experimentally validated here.

Federated scripts elsewhere in this repository are not evidence that these models were federated-trained, and no privacy, personalization, state-of-the-art, or accuracy-improvement claim is established by this run.

## Verified web preview

The interface is titled Multilingual Next-Word Prediction. It can preview a partially trained checkpoint with a visible warning, switches to the validation-selected checkpoint when available, and shows final accuracy only when test metrics exist. All-unknown input is rejected instead of displaying unconditioned guesses. Run `.\run_self_check.ps1 -RequireHealth` to repeat the current phase checks. Legacy multilingual self-check code remains historical.


## Completed baseline run

Both full-data one-epoch runs and held-out evaluations are complete. See [TELUGU_RESULTS.md](TELUGU_RESULTS.md) for measured results and checkpoint verification. The model selector, training/evaluation panel and experiment expander are hidden in the web interface at the user's request; LSTM remains the default.

