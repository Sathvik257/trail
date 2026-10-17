import sys
from pathlib import Path
sys.path.insert(0, str(Path('src').resolve()))
from streamlit.testing.v1 import AppTest
app = AppTest.from_file('src/telugu_app.py').run(timeout=30)
assert not app.exception, app.exception
assert app.radio[0].options == ['LSTM', 'Bi-LSTM']
app.button[0].click().run(timeout=30)
assert not app.exception, app.exception
assert app.info, 'Missing checkpoint should show training information'
app.radio[0].set_value('Bi-LSTM').run(timeout=30)
app.button[0].click().run(timeout=30)
assert not app.exception, app.exception
print('Telugu UI: both selectors and missing-checkpoint state passed')

