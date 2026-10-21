# CNN continuation results (2026-10-21)

Completed epochs 11–14 from CNN v2 epoch 10 using restored Adam, learning rate 0.0003, batch 512, unchanged splits/data hashes and validation top-1 selection with parent fallback. Selected epoch: 14. Checkpoint reload and finite inference passed both worker and independent checks. Original models and UI are preserved.

On 200,092 test examples: top-1 18.7409%, top-3 29.7478%, top-5 34.8320%, cross-entropy 5.221678, perplexity 185.2448. Relative to CNN v2, top-1 increased by 0.0635 percentage points and top-5 by 0.1274 points. Perplexity worsened from 183.5440. The longer-context hybrid remains higher at 20.8424% top-1.

Validation top-1 changed from 18.5650% (parent epoch 10) to 18.5821% (epoch 14), while cross-entropy worsened from 5.263112 to 5.273033. This is a small accuracy change with worse probabilistic loss, not evidence of a substantial gain or maximum achievable accuracy. No additional continuation is launched by this completed protocol.

For the research paper, report this single-seed result with its limitations. Repeated use of the same test set requires a fresh independent evaluation set for final confirmation; statistical significance and superiority are not established. Next evidence priorities are multi-seed replication and controlled gate ablation, with all selection on validation data.

Exact metrics and history: outputs/telugu_cnn_accuracy_v3. Source: src/continue_telugu_cnn_accuracy.py. Original protocol: docs/CNN_CONTINUATION_PROTOCOL.md.
