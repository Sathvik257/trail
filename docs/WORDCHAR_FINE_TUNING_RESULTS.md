# Gated word-character CNN fine-tuning (2026-10-26)

Completed epochs 6–10 from the original epoch-5 checkpoint, restoring Adam at learning rate 0.0003 and batch size 512. Maximum epoch 10, patience 2, selection by validation top-1 including parent fallback. Data hashes and surface lookup tensors were checked against the parent. Surface IDs, not vocabulary word IDs, feed the model. Baseline checkpoints and UI are preserved.

Validation selected epoch 9: top-1 19.4050%, cross-entropy 5.104256. Epoch 10 declined to 19.3497% with cross-entropy 5.109135. Epoch 6 cross-entropy was 5.097840: improving top-1 did not consistently improve probabilistic loss. No automatic further extension is justified by the final epoch's decline.

Full held-out evaluation: 200,092 examples; top-1 19.5995%, top-3 30.8703%, top-5 36.0649%, cross-entropy 5.056457, vocabulary perplexity 157.0332. Original word-character model: top-1 18.5525%, top-5 34.6780%, perplexity 167.1261. Gains are 1.0470 and 1.3869 percentage points respectively. The hybrid remains higher at 20.8424% top-1. OOV targets remain counted as incorrect; target OOV rate 7.2077%.

Worker verification passed checkpoint reload and finite inference after full held-out evaluation. An independent reload and surface-ID smoke inference also returned finite logits of shape [2, 20000]. Exact metrics, selection history, configuration, verification and best checkpoint are in outputs/telugu_wordchar_accuracy_v2. Resume checkpoints and runtime logs are excluded from publication.

Reproduction: after preparing the original data and parent checkpoint, run `python src/improve_telugu_wordchar_accuracy.py`. It verifies data hashes, resumes compatible checkpoints, and selects only on validation. Historical machine paths in experiment.json describe the original run.

Research limitations: single seed; repeated examination of the same test set across experiments; no statistical significance or maximum accuracy claim. A fresh independent test set, multiple seeds, and controlled gate ablations are needed before strong superiority or novelty claims. Future experiment selection must use validation, not these test results.
