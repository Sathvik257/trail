# Gated word-character CNN experiment

Five epochs, seed 42, Adam 0.001, batch 512, gradient clipping 1.0. Same raw
train/validation/test partitions, target vocabulary and eight-word contexts as
the baselines. Separate output: outputs/telugu_wordchar.

Character alphabet is fitted on training text only (pad=0, unknown character=1).
Each surface word gets a deterministic character lookup row, preserving OOV
input spelling. Held-out lookup rows are input encoding only, not learned word
embeddings and not training examples. Test targets still map to the original
word vocabulary. Characters are NFC Unicode code points, not morphemes or
grapheme clusters; words longer than 32 code points retain their last 32.

16-dimensional character embeddings -> kernel-3 character convolution -> masked
max pooling -> 64-dimensional character vector. A sigmoid gate over concatenated
word/character vectors mixes them element-wise. The existing three-layer causal
CNN then predicts the next word. Gate training is end-to-end; no morphology
labels are used. Character features are computed once per unique word per batch.

The standard top-k and fixed-vocabulary perplexity evaluation is unchanged.
Best checkpoint is chosen by validation loss. The web app default is unchanged.
Deployment/inference integration for arbitrary new spellings remains separate.
The completed run improves top-1 accuracy over word CNN from 17.67% to 18.55%. See FINAL_MODEL_RESULTS.md. Overall gains do not establish rare-word gains or algorithmic originality.
Ungated fusion ablation, training-frequency evaluation and multiple seeds remain
necessary to assess the proposed contribution.

Run: python src/complete_telugu_wordchar.py
Test: python src/check_wordchar.py
