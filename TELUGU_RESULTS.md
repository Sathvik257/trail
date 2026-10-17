# Telugu baseline training results

Both models completed 5 full epoch(s) over all 3,483,073 training examples. This completes the configured run; it does not establish convergence.

Validation: 198,928 examples. Test: 200,092 examples, using the validation-selected checkpoint. Unknown targets count as misses in word accuracy.

| Model | Top-1 | Top-3 | Top-5 | Vocabulary perplexity |
|---|---:|---:|---:|---:|
| LSTM | 18.03% | 28.78% | 33.57% | 199.02 |
| Bi-LSTM | 18.82% | 29.79% | 35.00% | 178.60 |

Bi-LSTM uses more parameters, so this is an equal-data/equal-epoch comparison, not a parameter-matched comparison. Perplexity collapses unknown words to the unknown token.

Both saved checkpoints were loaded successfully, verified to contain finite weights, and exercised on Telugu input. The web interface continues to use LSTM, with the model selector and training/evaluation panels hidden as requested.
