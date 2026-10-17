# Telugu frequency and unknown-input evaluation

Post-hoc diagnostic on the existing held-out test set, using the unchanged validation-selected checkpoints. Frequencies count target positions in training text only. No models were fitted or tuned.

| Group | Test examples | Word CNN top-1 | Character CNN top-1 | Word CNN top-5 | Character CNN top-5 |
|---|---:|---:|---:|---:|---:|
| all | 200,092 | 17.67% | 18.55% | 33.53% | 34.68% |
| oov | 14,422 | 0.00% | 0.00% | 0.00% | 0.00% |
| rare_1_10 | 125 | 2.40% | 1.60% | 3.20% | 3.20% |
| medium_11_100 | 29,846 | 5.85% | 6.04% | 12.95% | 13.34% |
| frequent_101_plus | 155,699 | 21.58% | 22.68% | 40.61% | 42.01% |
| unknown_prefix | 57,089 | 13.13% | 14.64% | 26.10% | 28.88% |
| known_prefix | 143,003 | 19.48% | 20.11% | 36.50% | 36.99% |

Rare means a known-vocabulary target with at most 10 training target occurrences. Medium means 11–100, frequent means over 100. OOV targets form a separate group and count as misses. Unknown-prefix examples contain at least one input word outside the word vocabulary. Prefix groups overlap frequency groups and must not be added together.

The rare group contains only 125 examples, so it cannot support broad claims about Telugu morphology or rare-word performance. Unknown-input performance does not imply the model can generate unseen target words. These are descriptive single-seed results, not significance tests or gate ablations. Overall top-1 and top-5 accuracies reproduce the saved test metrics exactly.

Measured finding: for 57,089 examples containing an unknown input word, character CNN top-1 rises from 13.13% to 14.64% and top-5 from 26.10% to 28.88%. Rare-known-target top-1 falls from 3/125 to 2/125, while top-5 stays 4/125. The evidence supports improved performance on inputs containing unknown words, not a rare-target improvement claim.
