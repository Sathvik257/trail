# CNN fine-tuning results — 2026-10-18

Original word CNN epoch 5 was continued with restored Adam state, learning rate 0.0003, batch 512, maximum epoch 10 and patience 2. Validation top-1 selected epoch 10. Dataset hashes, splits and architecture were unchanged. An interrupted run resumed from epoch 10 batch 5500.

On 200,092 held-out examples: top-1 18.6774%, top-3 29.6494%, top-5 34.7045%, perplexity 183.5440. Original CNN top-1 was 17.6669% and top-5 33.5316%: gains of 1.0105 and 1.1729 percentage points. This does not exceed the longer-context hybrid's 20.8424% top-1.

Checkpoint reload and finite inference passed both the worker check and an independent reload check. The UI default and baseline checkpoints remain unchanged.

## Continuation assessment

Validation top-1 rose from 18.3820% at epoch 6 to 18.5650% at epoch 10, but validation cross-entropy worsened from 5.2526 to 5.2631. Gains are small and probability calibration is not improving. A continuation is possible, but is not launched automatically here: for the research paper, a separately specified multi-seed replication and gate ablation are more informative than claiming maximum accuracy from additional epochs. Any next experiment must be selected with validation only, with a fresh independent evaluation set for final confirmation. This decision uses validation trends, not test scores.

Exact metrics, history, configuration and verification are in outputs/telugu_cnn_accuracy_v2. Run src/improve_telugu_cnn_accuracy.py with the original prepared data and original CNN parent checkpoint to reproduce the experiment.
