"""Multilingual next-word prediction: saved neural baseline interface."""
from pathlib import Path
import streamlit as st
from telugu_neural_predictor import TeluguNeuralPredictor
from text_utils import project_root

OUTPUT = project_root() / 'outputs' / 'telugu_comparison'


@st.cache_resource(max_entries=2)
def load_predictor(path, modified_ns, counts_path=None, counts_modified_ns=None):
    if counts_path:
        from telugu_hybrid_predictor import TeluguHybridPredictor
        return TeluguHybridPredictor(path, Path(counts_path))
    return TeluguNeuralPredictor(path)


def main():
    st.set_page_config(page_title='Multilingual Next-Word Prediction', page_icon='✍️', layout='centered')
    st.markdown('**LANGUAGE INTELLIGENCE · RESEARCH PROTOTYPE**')
    st.title('Multilingual Next-Word Prediction')
    st.caption('Predict the next word in Telugu, Hindi or English.')
    language = st.selectbox('Language', ['Telugu', 'Hindi', 'English']).lower()
    output = OUTPUT if language == 'telugu' else project_root() / 'outputs' / (language + '_hf_comparison')
    key = 'lstm'  # Model selection is hidden until requested.
    best = output / f'{key}_best.pt'
    latest = output / f'{key}_latest.pt'
    checkpoint = best if best.exists() else latest
    counts = None
    if language == 'telugu' and st.query_params.get('model') == 'hybrid':
        checkpoint = project_root() / 'outputs/telugu_accuracy_v2/bilstm_best.pt'
        counts = project_root() / 'outputs/telugu_hybrid_v2/counts.pkl'
        st.caption('Telugu hybrid · Held-out exact-word accuracy 20.84%')
    preview = counts is None and not best.exists() and latest.exists()
    if preview:
        st.warning('Training preview: this model is still learning. These are provisional predictions, not final evaluated results.')
    elif not checkpoint.exists():
        st.error('The selected model checkpoint is missing. Restore the published outputs folder or train this language first.')
    if counts is not None and not counts.exists():
        st.error('The Telugu hybrid count artifact is missing. Restore outputs/telugu_hybrid_v2/counts.pkl.')
    examples = {'telugu': 'ఆరోగ్యంగా ఉండటానికి', 'hindi': 'स्वस्थ रहने के लिए', 'english': 'To stay healthy'}
    text = st.text_area('Your phrase', value=examples[language], height=120,
                        placeholder='Type a few words…', max_chars=2000, key='input_' + language)
    if st.button('Predict next word', type='primary', disabled=not checkpoint.exists() or (counts is not None and not counts.exists()), use_container_width=True):
        try:
            predictor = load_predictor(str(checkpoint), checkpoint.stat().st_mtime_ns,
                                       str(counts) if counts else None, counts.stat().st_mtime_ns if counts else None)
            result = predictor.predict(text)
            if not result['context']:
                st.warning('Please enter a few words in the selected language.')
            elif not result['predictions']:
                st.warning('These words are outside the learned vocabulary. Try a phrase containing a familiar word.')
            else:
                if result['unknown_words']:
                    st.caption('Words outside the vocabulary: ' + ', '.join(result['unknown_words']))
                st.subheader('Next words')
                for rank, item in enumerate(result['predictions'], 1):
                    with st.container(border=True):
                        st.markdown(f"**{rank}. {item['word']}**")
                        st.caption(f"Model probability · {item['probability']:.1%}")
                st.caption('Probabilities are model estimates, not guarantees of correctness.')
        except (OSError, RuntimeError, ValueError, KeyError):
            st.error('The model could not be loaded. Check that the published model artifacts are intact.')
            import logging
            logging.getLogger(__name__).exception('Prediction failed')


if __name__ == '__main__':
    main()
