# Evaluation results

Snapshot of completed saved experiments, reviewed 24 November 2026. Values below are read from the linked JSON files; cleanup did not retrain or reevaluate the full datasets.

## Complete experiment ledger

| Experiment / metric file | Top-1 | Top-5 | Vocabulary perplexity | Test examples |
|---|---:|---:|---:|---:|
| [english_hf_bilstm_continuation/bilstm_test_metrics](../outputs/english_hf_bilstm_continuation/bilstm_test_metrics.json) | 27.23% | 47.32% | 85.82 | 308,612 |
| [english_hf_comparison/bilstm_test_metrics](../outputs/english_hf_comparison/bilstm_test_metrics.json) | 26.59% | 46.62% | 88.99 | 308,612 |
| [english_hf_comparison/lstm_test_metrics](../outputs/english_hf_comparison/lstm_test_metrics.json) | 25.65% | 44.74% | 102.91 | 308,612 |
| [english_hf_hybrid_v1/test_metrics](../outputs/english_hf_hybrid_v1/test_metrics.json) | 29.85% | 50.63% | — | 308,612 |
| [english_hf_hybrid_v2/test_metrics](../outputs/english_hf_hybrid_v2/test_metrics.json) | 29.85% | 50.63% | — | 308,612 |
| [hindi_discounted_hybrid_v1/test_metrics](../outputs/hindi_discounted_hybrid_v1/test_metrics.json) | 32.79% | 56.43% | — | 284,823 |
| [hindi_hf_bilstm_continuation/bilstm_test_metrics](../outputs/hindi_hf_bilstm_continuation/bilstm_test_metrics.json) | 30.89% | 54.40% | 51.02 | 284,823 |
| [hindi_hf_comparison/bilstm_test_metrics](../outputs/hindi_hf_comparison/bilstm_test_metrics.json) | 30.49% | 53.93% | 52.50 | 284,823 |
| [hindi_hf_comparison/lstm_test_metrics](../outputs/hindi_hf_comparison/lstm_test_metrics.json) | 29.41% | 52.32% | 60.05 | 284,823 |
| [hindi_hf_hybrid_v1/test_metrics](../outputs/hindi_hf_hybrid_v1/test_metrics.json) | 32.74% | 56.14% | — | 284,823 |
| [hindi_hf_hybrid_v2/test_metrics](../outputs/hindi_hf_hybrid_v2/test_metrics.json) | 32.79% | 56.43% | — | 284,823 |
| [telugu_accuracy_v2/bilstm_test_metrics](../outputs/telugu_accuracy_v2/bilstm_test_metrics.json) | 19.42% | 35.56% | 176.04 | 200,092 |
| [telugu_accuracy_v3/bilstm_test_metrics](../outputs/telugu_accuracy_v3/bilstm_test_metrics.json) | 19.35% | 35.72% | 169.48 | 200,092 |
| [telugu_cnn/cnn_test_metrics](../outputs/telugu_cnn/cnn_test_metrics.json) | 17.67% | 33.53% | 193.76 | 200,092 |
| [telugu_cnn_accuracy_v2/cnn_test_metrics](../outputs/telugu_cnn_accuracy_v2/cnn_test_metrics.json) | 18.68% | 34.70% | 183.54 | 200,092 |
| [telugu_cnn_accuracy_v3/cnn_test_metrics](../outputs/telugu_cnn_accuracy_v3/cnn_test_metrics.json) | 18.74% | 34.83% | 185.24 | 200,092 |
| [telugu_comparison/bilstm_test_metrics](../outputs/telugu_comparison/bilstm_test_metrics.json) | 18.82% | 35.00% | 178.60 | 200,092 |
| [telugu_comparison/lstm_test_metrics](../outputs/telugu_comparison/lstm_test_metrics.json) | 18.03% | 33.57% | 199.02 | 200,092 |
| [telugu_discounted_hybrid_v1/test_metrics](../outputs/telugu_discounted_hybrid_v1/test_metrics.json) | 20.85% | 37.84% | — | 200,092 |
| [telugu_hybrid_v1/test_metrics](../outputs/telugu_hybrid_v1/test_metrics.json) | 20.62% | 37.63% | — | 200,092 |
| [telugu_hybrid_v2/test_metrics](../outputs/telugu_hybrid_v2/test_metrics.json) | 20.84% | 37.46% | — | 200,092 |
| [telugu_wordchar/wordchar_cnn_test_metrics](../outputs/telugu_wordchar/wordchar_cnn_test_metrics.json) | 18.55% | 34.68% | 167.13 | 200,092 |
| [telugu_wordchar_accuracy_v2/wordchar_cnn_test_metrics](../outputs/telugu_wordchar_accuracy_v2/wordchar_cnn_test_metrics.json) | 19.60% | 36.06% | 157.03 | 200,092 |

## Interpretation and model selection

Baseline models select minimum validation cross-entropy. Continuations and hybrids select validation exact-word accuracy. Hindi continuation selected epoch 6; English selected epoch 8. Telugu Bi-LSTM v2 and word-character CNN v2 selected epoch 9. Perplexity collapses unseen targets into the unknown token and is not an open-vocabulary measure. Hybrid perplexity was not measured.

Telugu hybrid v2 mixes 50% continued Bi-LSTM and 50% long-context counts. Hindi hybrid v2 selected a 35% count weight; English retained 50%. The Hindi v2 gain over v1 was 0.0502 percentage points. The discounted follow-up selected a new Telugu candidate (a 0.0075-point top-1 gain) and retained Hindi's parent; English's partial evaluation is excluded. Small single-seed gains are not demonstrated statistical significance.

The [Telugu count-only diagnostic](../outputs/telugu_count_only/evaluation.json) isolates the count component. Neural weights and counts were fitted using training data; selection records, histories, verification receipts and data hashes are kept beside the metrics. Original absolute paths in historical JSON describe the run machine, not installation requirements.

## Demo versus experiments

The UI defaults to `outputs/{telugu_comparison,hindi_hf_comparison,english_hf_comparison}/lstm_best.pt`. The Telugu `?model=hybrid` route uses `telugu_accuracy_v2/bilstm_best.pt` and `telugu_hybrid_v2/counts.pkl`. The newer discounted candidate is not deployed in that route. Hindi/English hybrid counts are retained in their `*_hf_hybrid_v1` directories; v2 selection reuses them.

## Limits

Each language has its own corpus. Do not rank language difficulty using these percentages. Fixed-vocabulary exact-word accuracy credits one target even when several next words would be valid. Unknown targets count as misses. The repeated test reporting, one seed, possible semantic duplication and synthetic/translated domain limit generalization claims. No independent evaluation, confidence intervals or novelty claim is established by these records.
