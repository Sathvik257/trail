"""Check the live Telugu interface and real saved checkpoint."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json
import torch
from streamlit.testing.v1 import AppTest
from telugu_neural_predictor import TeluguNeuralPredictor
from text_utils import project_root

root = project_root()
out = root / 'outputs' / 'telugu_comparison'
path = out / 'lstm_best.pt'
if not path.exists():
    path = out / 'lstm_latest.pt'
predictor = TeluguNeuralPredictor(path)
assert predictor.predict('')['predictions'] == []
assert predictor.predict('hello 123 !!!')['predictions'] == []
unknown = 'ఔఔఔఔఔఔఔఔఔఔ'
assert unknown not in predictor.vocab
assert predictor.predict(unknown)['predictions'] == []
result = predictor.predict('ఆరోగ్యంగా ఉండటానికి')
assert len(result['predictions']) == 5
assert len({r['word'] for r in result['predictions']}) == 5
assert all(0 <= r['probability'] <= 1 for r in result['predictions'])
assert all(r['word'] not in ('<pad>', '<unk>') for r in result['predictions'])
assert all(torch.isfinite(t).all() for t in predictor.model.state_dict().values())
app = AppTest.from_file(str(root / 'src' / 'telugu_app.py')).run(timeout=30)
assert not app.exception, app.exception
assert 'Multilingual' in app.title[0].value
app.button[0].click().run(timeout=30)
assert not app.exception, app.exception
assert any('తదుపరి పదాలు' in s.value for s in app.subheader)
for text in ['', 'hello', unknown]:
    app.text_area[0].set_value(text)
    app.button[0].click().run(timeout=30)
    assert not app.exception, app.exception
    assert app.warning
report = dict(status='passed', checkpoint=str(path), architecture=predictor.checkpoint['architecture'],
              epoch=predictor.checkpoint['epoch'], partial_batch=predictor.checkpoint['partial_batch'],
              checks=['finite checkpoint weights', 'real Telugu predictions', 'no special-token suggestions',
                      'empty and non-Telugu input', 'all unknown input', 'multilingual branding',
                      'LSTM UI prediction', 'default LSTM interface'])
(root / 'outputs' / 'telugu_ui_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))

