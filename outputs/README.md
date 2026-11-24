# Saved experiment artifacts

Each directory represents one experiment. `*_best.pt` contains a selected neural checkpoint (including vocabulary and optimizer state); `counts.pkl` contains fitted training-only counts. JSON files retain metrics, configuration, selection, histories and verification evidence. See [the results ledger](../docs/RESULTS.md) for the current interpretation.

The default demo loads `lstm_best.pt` from each language's baseline comparison directory. Telugu hybrid v2 loads its continued Bi-LSTM and v2 counts. Hindi/English v2 reuses counts from v1. Do not rename artifact paths without updating consumers and documentation.

Raw datasets, runtime logs, duplicate `*_latest.pt`, temporary locks and generated `checks/` receipts are excluded. Historical status files are snapshots; they do not indicate live jobs. Original machine paths in receipts are informational.
