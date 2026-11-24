"""Bounded Hindi and English Bi-LSTM continuation from validated baselines."""
import json
from improve_telugu_accuracy import ROOT, fine_tune
from train_language_comparison import audit_data

if __name__ == "__main__":
    for language in ("hindi", "english"):
        parent_dir = ROOT / "outputs" / f"{language}_hf_comparison"
        out = ROOT / "outputs" / f"{language}_hf_bilstm_continuation"
        status = out / "status.json"
        if status.exists() and json.loads(status.read_text()).get("verification_passed"):
            continue
        fine_tune(ROOT / f"data_{language}_hf", parent_dir / "bilstm_best.pt",
                  out, audit=audit_data, maximum_epoch=8)
