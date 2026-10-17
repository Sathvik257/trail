"""Run the isolated five-epoch CNN experiment and verify the saved model."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from model import model_from_checkpoint
from train_telugu_comparison import save_json

root = Path(__file__).resolve().parents[1]
output = root / 'outputs' / 'telugu_wordchar'
output.mkdir(parents=True, exist_ok=True)
awake = False
try:
    if os.name == 'nt':
        awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    subprocess.run([sys.executable, '-X', 'utf8', '-u', str(root / 'src/train_telugu_comparison.py'),
                    '--wordchar-only', '--output-dir', str(output), '--epochs', '5', '--threads', '4'],
                   cwd=root, check=True)
    torch.set_num_threads(1)
    latest = torch.load(output / 'wordchar_cnn_latest.pt', map_location='cpu', weights_only=True)
    best = torch.load(output / 'wordchar_cnn_best.pt', map_location='cpu', weights_only=True)
    metrics = json.loads((output / 'wordchar_cnn_test_metrics.json').read_text())
    assert latest['epoch'] == 5 and latest['partial_batch'] == 0
    assert metrics['completed_epochs'] == 5
    model = model_from_checkpoint(best).eval()
    inputs = torch.tensor([[0] * 6 + [2, 3], [2] * 8])
    with torch.inference_mode():
        assert torch.isfinite(model(inputs)).all()
    save_json(output / 'verification.json', dict(status='passed', completed_epochs=5,
              best_epoch=best['epoch'], parameter_count=sum(p.numel() for p in model.parameters()),
              test_metrics=metrics))
    print('Word-character CNN training, held-out evaluation and checkpoint verification complete.', flush=True)
finally:
    if awake:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

