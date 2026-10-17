"""Create a portable project snapshot, including prepared data and checkpoints."""
import json
import zipfile
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parents[1]
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
archive = root / f'Multilingual_Next_Word_Prediction_{stamp}.zip'
folders = ['src', 'data', 'data_telugu_llm', 'data_hindi', 'data_english', 'outputs']
files = [p for p in root.iterdir() if p.is_file() and (p.suffix.lower() in {'.md', '.ps1', '.bat'} or p.name in {'requirements.txt', '.gitignore'})]
for folder in folders:
    files.extend(p for p in (root / folder).rglob('*') if p.is_file())
files = sorted(set(p for p in files if '__pycache__' not in p.parts and p.suffix.lower() not in {'.pyc', '.pyo', '.tmp', '.pid', '.log', '.zip'} and p.name != '.training.lock'))
notes = '''MULTILINGUAL NEXT-WORD PREDICTION - PROJECT SNAPSHOT

Current implementation: Telugu LSTM and Bi-LSTM. Further training may still be
running on the original computer. This ZIP contains checkpoints available when
it was created; it is not a claim that five-epoch training has completed.

Windows setup (PowerShell, from this extracted folder):
  py -m venv .venv
  .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
  .\\run_app.ps1

Resume training locally, if wanted:
  .\\run_telugu_comparison.ps1 -Epochs 5 -Threads 4 -FirstModel bilstm
Do not run two trainers in the same extracted directory.
The application uses LSTM by default; model and training panels are hidden.

Includes source, prepared datasets, available model checkpoints, documentation,
and launch scripts. Python environments, Git metadata, temporary files and
process IDs are excluded. Files in outputs are snapshots, not live status on
the recipient computer. Older report figures are historical; see
TELUGU_EXPERIMENT.md and model-specific test metrics for experiment details.

Dataset attribution and recorded licenses: data_telugu_llm/provenance.json.
This package has not been tested on a fresh second computer.
'''
manifest = dict(created_local=datetime.now().astimezone().isoformat(), snapshot=True, files=[])
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as z:
    z.writestr('START_HERE.txt', notes)
    for index, path in enumerate(files, 1):
        relative = path.relative_to(root).as_posix()
        if path.suffix in {'.pt', '.json'}:
            # Capture one complete version of atomically replaced training files.
            content = path.read_bytes()
            z.writestr(relative, content)
            size = len(content)
        else:
            z.write(path, relative)
            size = path.stat().st_size
        manifest['files'].append(dict(path=relative, bytes=size))
        if index % 50 == 0:
            print(f'Packaged {index}/{len(files)} files', flush=True)
    z.writestr('PACKAGE_MANIFEST.json', json.dumps(manifest, indent=2))
print('Checking archive integrity...', flush=True)
with zipfile.ZipFile(archive) as z:
    bad = z.testzip()
    if bad:
        raise RuntimeError(f'Archive checksum failed: {bad}')
    required = {'src/telugu_app.py', 'src/model.py', 'data_telugu_llm/vocab.pkl', 'outputs/telugu_comparison/lstm_best.pt', 'outputs/telugu_comparison/bilstm_best.pt', 'run_telugu_comparison.ps1', 'START_HERE.txt'}
    assert required.issubset(z.namelist()), 'Missing required release files'
print(json.dumps(dict(archive=str(archive), size_mb=round(archive.stat().st_size / 1024**2, 1), files=len(files)+2, integrity='passed')), flush=True)
