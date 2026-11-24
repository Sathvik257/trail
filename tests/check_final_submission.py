import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

"""Exercise actual saved models and multilingual Streamlit user flows."""
import json
from datetime import datetime, timezone
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
from telugu_neural_predictor import TeluguNeuralPredictor

def main():
    receipts = []
    for language in ('Telugu', 'Hindi', 'English'):
        app = AppTest.from_file(str(ROOT / 'src/app.py')).run(timeout=60)
        app.selectbox[0].select(language).run(timeout=60)
        assert not app.exception, app.exception
        phrase = app.text_area[0].value
        app.button[0].click().run(timeout=60)
        assert not app.exception, app.exception
        assert any(x.value == 'Next words' for x in app.subheader)
        folder = 'telugu_comparison' if language == 'Telugu' else language.lower() + '_hf_comparison'
        cp = ROOT / 'outputs' / folder / 'lstm_best.pt'
        predictor = TeluguNeuralPredictor(cp)
        result = predictor.predict(phrase)
        assert len(result['predictions']) == 5
        assert len(set(p['word'] for p in result['predictions'])) == 5
        assert all(0 <= p['probability'] <= 1 for p in result['predictions'])
        app.text_area[0].set_value('')
        app.button[0].click().run(timeout=60)
        assert not app.exception and app.warning
        receipts.append(dict(language=language, checkpoint=str(cp.relative_to(ROOT)),
                             phrase=phrase, result=result, empty_input_warning=True))
    app = AppTest.from_file(str(ROOT / 'src/app.py'))
    app.query_params['model'] = 'hybrid'
    app.run(timeout=60).button[0].click().run(timeout=60)
    assert not app.exception, app.exception
    assert any(x.value == 'Next words' for x in app.subheader)
    out = ROOT / 'outputs' / 'checks'
    out.mkdir(parents=True, exist_ok=True)
    report = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
                  languages=receipts, telugu_hybrid_ui=True,
                  scope='Real checkpoint inference and Streamlit AppTest; not a new full test-set evaluation')
    (out / 'demo_verification.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('PASS: three language prediction flows, empty input, and Telugu hybrid UI')

if __name__ == '__main__':
    main()
