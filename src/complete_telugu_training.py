"""Finish the configured training, prevent idle sleep during it, and verify outputs."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[1]
output = root / 'outputs' / 'telugu_comparison'
output.mkdir(parents=True, exist_ok=True)
result_path = output / 'completion_job.json'
def record(state, **details):
    temp = result_path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(dict(status=state, updated_utc=datetime.now(timezone.utc).isoformat(), **details), indent=2), encoding='utf-8')
    temp.replace(result_path)

awake = False
try:
    if os.name == 'nt':
        awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    record('running', idle_sleep_prevented=awake, target_epochs=5)
    print(f'Completing five epochs per model; temporary idle-sleep prevention: {awake}', flush=True)
    subprocess.run([sys.executable, '-X', 'utf8', '-u', str(root / 'src' / 'train_telugu_comparison.py'),
                    '--epochs', '5', '--threads', '4', '--first-model', 'bilstm'], cwd=root, check=True)
    record('verifying', target_epochs=5)
    subprocess.run([sys.executable, '-X', 'utf8', str(root / 'src' / 'verify_telugu_training.py')], cwd=root, check=True)
    subprocess.run([sys.executable, '-X', 'utf8', str(root / 'src' / 'check_telugu_ui.py')], cwd=root, check=True)
    record('complete', target_epochs=5, report=str(root / 'TELUGU_RESULTS.md'))
    print('Training, model verification, results report and interface checks completed.', flush=True)
except BaseException as exc:
    record('failed', error=str(exc))
    raise
finally:
    if awake:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
