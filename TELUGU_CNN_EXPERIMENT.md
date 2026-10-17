# Telugu CNN experiment

Implemented: word-level residual gated causal CNN. Embeddings 64; channels 128;
three kernel-3 layers with dilations 1, 2, 4; receptive field 15 positions, covering
the existing eight-word prefix. Left-only padding prevents future-token access.
The last position feeds a 20,000-class output. No target enters the prefix.

Run separately from the ongoing recurrent experiment:
`python src/train_telugu_comparison.py --cnn-only --output-dir outputs/telugu_cnn --epochs 5 --threads 4`

Same dataset, vocabulary, split hashes, seed 42, Adam 0.001, batch 512,
gradient clipping 1.0 and five-epoch budget. Best checkpoint selected only by
validation cross-entropy; held-out top-1/3/5, perplexity and OOV metrics follow.
Parameter counts differ; this is not a parameter-matched comparison. Concurrent
CPU jobs invalidate raw wall-time speed comparisons. Accuracy is unknown until
evaluation completes. The app default and hidden model controls are unchanged.

## Novelty boundary and research hypothesis

Gated convolutional language modeling is established, not a new algorithm:
https://proceedings.mlr.press/v70/dauphin17a.html

Do not claim first, best, state of the art, or superior accuracy from this run.
A defensible future hypothesis is that character/subword-aware representations
improve predictions for rare and inflected Telugu forms. That extension is not
implemented here. Word-ID inputs currently lose OOV spelling, and a fixed word
output cannot generate unseen words; adding a character encoder alone would not
solve output OOV coverage. It needs a revised input/data pipeline and, for unseen
outputs, subword generation with explicit whole-word evaluation.

Required evidence: word-CNN baseline versus the proposed extension, rare-word
frequency buckets computed from training only, equal context and data splits,
multiple seeds, parameter/latency measurements on isolated hardware, and ablations
of character features and gating. A broader literature review is needed before
asserting research originality. Completed LSTM and Bi-LSTM are reference baselines.
