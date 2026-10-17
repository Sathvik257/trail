"""Multilingual next-word prediction: current Telugu baseline interface."""
import json
from pathlib import Path
import streamlit as st
from telugu_neural_predictor import TeluguNeuralPredictor
from text_utils import project_root

OUTPUT = project_root() / 'outputs' / 'telugu_comparison'


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


@st.cache_resource(max_entries=2)
def load_predictor(path, modified_ns):
    return TeluguNeuralPredictor(path)


@st.fragment(run_every='10s')
def training_progress():
    status = read_json(OUTPUT / 'status.json')
    state = status.get('status')
    if state == 'training':
        name = 'Bi-LSTM' if status.get('architecture') == 'bilstm' else 'LSTM'
        fraction = min(1., status.get('batch', 0) / max(status.get('batches', 1), 1))
        st.progress(fraction, text=f"{name} · Epoch {status.get('epoch', 1)} · {fraction:.0%} of this epoch")
        st.caption('Training continues in the background. Progress refreshes every 10 seconds.')
    elif state == 'complete':
        st.success('Both baseline models have completed training and test evaluation.')
    elif state == 'failed':
        st.error('Training stopped. Check the training log before resuming.')
    elif state == 'evaluating_test':
        st.info(f"Evaluating {status.get('architecture', 'model').upper()} on the held-out test set.")
    else:
        st.info('Preparing training data.' if state else 'Training status is not available yet.')


def main():
    st.set_page_config(page_title='Multilingual Next-Word Prediction', page_icon='✍️', layout='centered')
    st.markdown('**LANGUAGE INTELLIGENCE · RESEARCH PROTOTYPE**')
    st.title('Multilingual Next-Word Prediction')
    st.caption('Predict the next word from a Telugu phrase. Additional languages are planned.')
    st.selectbox('Language', ['Telugu'])
    key = 'lstm'  # Model selection is hidden until requested.
    best = OUTPUT / f'{key}_best.pt'
    latest = OUTPUT / f'{key}_latest.pt'
    checkpoint = best if best.exists() else latest
    preview = not best.exists() and latest.exists()
    if preview:
        st.warning('Training preview: this model is still learning. These are provisional predictions, not final evaluated results.')
    elif not checkpoint.exists():
        st.info('Bi-LSTM is queued after LSTM.' if key == 'bilstm' else 'Waiting for the first saved checkpoint.')
    text = st.text_area('మీ తెలుగు వాక్యం', value='ఆరోగ్యంగా ఉండటానికి', height=120,
                        placeholder='Type a few words in Telugu…', max_chars=2000)
    st.caption('తెలుగు వాక్యం టైప్ చేయండి. తర్వాత వచ్చే పదాలను చూడండి.')
    if st.button('Predict next word', type='primary', disabled=not checkpoint.exists(), use_container_width=True):
        try:
            predictor = load_predictor(str(checkpoint), checkpoint.stat().st_mtime_ns)
            result = predictor.predict(text)
            if not result['context']:
                st.warning('Please enter a few words in Telugu script.')
            elif not result['predictions']:
                st.warning('These words are outside the learned vocabulary. Try a phrase containing a familiar Telugu word.')
            else:
                if result['unknown_words']:
                    st.caption('Words outside the vocabulary: ' + ', '.join(result['unknown_words']))
                st.subheader('తదుపరి పదాలు')
                for rank, item in enumerate(result['predictions'], 1):
                    with st.container(border=True):
                        st.markdown(f"**{rank}. {item['word']}**")
                        st.caption(f"Model probability · {item['probability']:.1%}")
                st.caption('Probabilities are model estimates, not guarantees of correctness.')
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            st.error('The model could not be loaded. Try again after the next checkpoint is saved.')
            import logging
            logging.getLogger(__name__).exception('Prediction failed')


if __name__ == '__main__':
    main()



