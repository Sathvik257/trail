# Final Telugu model results

All four models completed five epochs. Checkpoint verification passed, and the LSTM web interface passed input-handling and prediction checks. Completion means the configured experiment finished, not that convergence or optimal accuracy is established.

Train: 3,483,073 examples. Validation: 198,928. Held-out test: 200,092. Vocabulary: 20,000 including special tokens. Context: eight words. Seed: 42. Checkpoints selected by lowest validation loss.

| Model | Epochs | Top-1 | Top-3 | Top-5 | Perplexity |
|---|---:|---:|---:|---:|---:|
| LSTM | 5 | 18.03% | 28.78% | 33.57% | 199.02 |
| Bi-LSTM | 5 | 18.82% | 29.79% | 35.00% | 178.60 |
| Word CNN | 5 | 17.67% | 28.42% | 33.53% | 193.76 |
| Gated word-character CNN | 5 | 18.55% | 29.59% | 34.68% | 167.13 |

The character CNN improves over word CNN by 0.89 percentage points in top-1 accuracy and 1.15 points in top-5 accuracy. Bi-LSTM has the highest accuracy, while the character CNN has the lowest fixed-vocabulary perplexity. Unknown targets (7.21%) count as accuracy misses. Perplexity collapses unknown words to UNK.

These are single-seed, unequal-parameter comparisons. They do not prove algorithmic originality or isolate the fusion gate's contribution. An independently trained ungated-fusion ablation and multiple seeds remain future experiments. Additional languages and arbitrary-new-spelling character-model deployment remain outside the current verified interface.

Evidence: outputs/telugu_comparison/*_test_metrics.json, outputs/telugu_cnn/cnn_test_metrics.json, outputs/telugu_wordchar/wordchar_cnn_test_metrics.json. Verification receipts are stored in the corresponding output directories and outputs/telugu_ui_check.json.

Additional diagnostic: see [TELUGU_FREQUENCY_RESULTS.md](TELUGU_FREQUENCY_RESULTS.md). Both CNN overall metrics reproduce exactly. Unknown-prefix top-1 improves from 13.13% to 14.64%; rare-known-target improvement is not established.
