"""Verify completed Telugu training artifacts and write the measured results."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import torch
from telugu_neural_predictor import TeluguNeuralPredictor
from text_utils import project_root

root = project_root()
out = root / 'outputs' / 'telugu_comparison'
status = json.loads((out / 'status.json').read_text(encoding='utf-8'))
if status['status'] != 'complete':
    raise SystemExit('Training/evaluation is not complete yet.')
comparison = json.loads((out / 'comparison.json').read_text(encoding='utf-8'))
verified = {}
rows = []
for key, name in [('lstm', 'LSTM'), ('bilstm', 'Bi-LSTM')]:
    metrics = comparison[key]
    path = out / f'{key}_best.pt'
    predictor = TeluguNeuralPredictor(path)
    checkpoint = predictor.checkpoint
    assert checkpoint['architecture'] == key
    assert checkpoint['partial_batch'] == 0
    assert metrics['completed_epochs'] == status['requested_epochs']
    assert checkpoint['epoch'] == metrics['best_epoch']
    assert metrics['training_examples_per_epoch'] == 3483073
    assert metrics['examples'] == 200092
    assert checkpoint['experiment']['validation_examples'] == 198928
    assert all(torch.isfinite(tensor).all() for tensor in predictor.model.state_dict().values())
    assert 0 <= metrics['top1_accuracy'] <= metrics['top3_accuracy'] <= metrics['top5_accuracy'] <= 1
    predictions = predictor.predict('ఆరోగ్యంగా ఉండటానికి')['predictions']
    assert len(predictions) == 5
    assert all(item['word'] not in ('<pad>', '<unk>') for item in predictions)
    verified[key] = dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         epoch=checkpoint['epoch'], prediction_count=len(predictions), status='passed')
    rows.append(f"| {name} | {metrics['top1_accuracy']:.2%} | {metrics['top3_accuracy']:.2%} | {metrics['top5_accuracy']:.2%} | {metrics['vocabulary_perplexity']:.2f} |")
report = '# Telugu baseline training results\n\n'
report += f"Both models completed {status['requested_epochs']} full epoch(s) over all 3,483,073 training examples. This completes the configured run; it does not establish convergence.\n\n"
report += 'Validation: 198,928 examples. Test: 200,092 examples, using the validation-selected checkpoint. Unknown targets count as misses in word accuracy.\n\n'
report += '| Model | Top-1 | Top-3 | Top-5 | Vocabulary perplexity |\n|---|---:|---:|---:|---:|\n' + '\n'.join(rows)
report += '\n\nBi-LSTM uses more parameters, so this is an equal-data/equal-epoch comparison, not a parameter-matched comparison. Perplexity collapses unknown words to the unknown token.\n\n'
report += 'Both saved checkpoints were loaded successfully, verified to contain finite weights, and exercised on Telugu input. The web interface continues to use LSTM, with the model selector and training/evaluation panels hidden as requested.\n'
(root / 'TELUGU_RESULTS.md').write_text(report, encoding='utf-8')
(out / 'completion_verification.json').write_text(json.dumps(dict(verified_utc=datetime.now(timezone.utc).isoformat(), models=verified), indent=2), encoding='utf-8')
print(report)

