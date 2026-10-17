from __future__ import annotations

import html

import streamlit as st

from agentic_predictor import AgenticPredictionService
from text_utils import LANGUAGE_RANGES, project_root


ROOT = project_root()
DEFAULT_TOP_K = 5
DEFAULT_SENTENCE_COUNT = 5
LANGUAGE_OPTIONS = {
    "Telugu": {
        "code": "telugu",
        "title": "Telugu Word and Sentence Prediction",
        "hint": "మీ తెలుగు వాక్యాన్ని టైప్ చేయండి. మోడల్ తర్వాత వచ్చే 5 పదాలు మరియు 5 వాక్య సూచనలు చూపిస్తుంది.",
        "label": "మీ తెలుగు వాక్యం",
        "placeholder": "ఉదాహరణ: ఆరోగ్యంగా ఉండటానికి",
        "default": "ఆరోగ్యంగా ఉండటానికి",
        "output": "తదుపరి పదాలు",
        "sentence_output": "వాక్య సూచనలు",
        "empty": "సరైన తెలుగు పదాలు దొరకలేదు. ఇంకొన్ని తెలుగు పదాలు టైప్ చేసి ప్రయత్నించండి.",
        "fallback": ["చేస్తుంది", "చూపిస్తుంది", "అవుతుంది", "ఉంటుంది", "పెరుగుతుంది"],
    },
    "Hindi": {
        "code": "hindi",
        "title": "Hindi Word and Sentence Prediction",
        "hint": "हिंदी वाक्य लिखें। मॉडल अगले 5 शब्द और 5 वाक्य सुझाव दिखाएगा।",
        "label": "आपका हिंदी वाक्य",
        "placeholder": "उदाहरण: सर्वर हर राउंड के बाद नया मॉडल",
        "default": "मैं हिंदी",
        "output": "अगले शब्द",
        "sentence_output": "वाक्य सुझाव",
        "empty": "सही हिंदी शब्द नहीं मिले। कुछ और हिंदी शब्द लिखकर प्रयास करें।",
        "fallback": ["करता", "दिखाता", "होता", "बढ़ता", "सीखता"],
    },
    "English": {
        "code": "english",
        "title": "English Word and Sentence Prediction",
        "hint": "Type an English sentence. The model will show 5 next words and 5 sentence suggestions.",
        "label": "Your English sentence",
        "placeholder": "Example: federated learning protects",
        "default": "I am learning English",
        "output": "Next words",
        "sentence_output": "Sentence suggestions",
        "empty": "No useful English words were found. Try a few more English words.",
        "fallback": ["helps", "improves", "predicts", "learns", "works"],
    },
}


@st.cache_resource
def get_service(language: str) -> AgenticPredictionService:
    return AgenticPredictionService(ROOT, language=language)


def has_language_char(text: str, language: str) -> bool:
    start, end = LANGUAGE_RANGES[language]
    return any(start <= char <= end for char in text)


def filter_predictions(words: list[str], text: str, language: str) -> list[str]:
    blocked_words = set(text.split())
    filled: list[str] = []
    for word in words:
        if not has_language_char(word, language):
            continue
        if word in blocked_words or word in filled:
            continue
        filled.append(word)
        if len(filled) == DEFAULT_TOP_K:
            return filled
    return filled


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 900px;
            padding-top: 2rem;
        }
        h1 {
            letter-spacing: 0;
            margin-bottom: .25rem;
        }
        div[data-testid="stTextArea"] textarea {
            min-height: 250px !important;
            font-size: 1.45rem !important;
            line-height: 1.8 !important;
            border-radius: 8px !important;
        }
        .stButton > button {
            min-height: 3.2rem;
            font-size: 1.1rem;
            font-weight: 700;
            border-radius: 8px;
        }
        .output-title {
            margin-top: 1.4rem;
            margin-bottom: .55rem;
            font-size: 1.15rem;
            font-weight: 700;
            color: inherit;
        }
        .prediction-row {
            border: 1px solid #cfe0d5;
            background: #f4fbf6;
            border-radius: 8px;
            color: #123b2a;
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.35;
            margin-bottom: .75rem;
            padding: .95rem 1.1rem;
        }
        .sentence-row {
            border: 1px solid #d8dde7;
            background: #f8fafc;
            border-radius: 8px;
            color: #1f2937;
            font-size: 1.18rem;
            font-weight: 650;
            line-height: 1.55;
            margin-bottom: .75rem;
            padding: .95rem 1.1rem;
        }
        .prediction-rank {
            color: #61766b;
            display: inline-block;
            font-size: 1.05rem;
            font-weight: 700;
            min-width: 2.2rem;
            vertical-align: middle;
        }
        .hint {
            color: #61766b;
            font-size: .95rem;
        }
        @media (max-width: 680px) {
            div[data-testid="stTextArea"] textarea {
                font-size: 1.2rem !important;
                min-height: 220px !important;
            }
            .prediction-row {
                font-size: 1.55rem;
            }
            .sentence-row {
                font-size: 1.05rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_predictions(text: str, language: str, options: dict[str, object]) -> None:
    service = get_service(language)
    result = service.predict(text, top_k=DEFAULT_TOP_K, mode="auto")
    predictions = filter_predictions(
        result.predictions,
        result.input_text,
        language,
    )

    st.markdown(f"<div class='output-title'>{options['output']}</div>", unsafe_allow_html=True)
    if not predictions:
        st.warning("No supported next-word suggestions for this input. The current training vocabulary is limited. Try adding more context or use a phrase from the training data.")
        return

    for rank, word in enumerate(predictions[:DEFAULT_TOP_K], start=1):
        st.markdown(
            f"<div class='prediction-row'><span class='prediction-rank'>{rank}</span>{html.escape(word)}</div>",
            unsafe_allow_html=True,
        )

    st.caption(result.explanation)

    sentence_result = service.predict_sentences(
        text,
        count=DEFAULT_SENTENCE_COUNT,
        max_new_words=8,
        mode="auto",
    )
    sentences = [
        sentence
        for sentence in sentence_result.predictions
        if has_language_char(sentence, language)
    ][:DEFAULT_SENTENCE_COUNT]
    if not sentences:
        return

    st.markdown(f"<div class='output-title'>{options['sentence_output']}</div>", unsafe_allow_html=True)
    for rank, sentence in enumerate(sentences, start=1):
        st.markdown(
            f"<div class='sentence-row'><span class='prediction-rank'>{rank}</span>{html.escape(sentence)}</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Indic Next Word Prediction",
        page_icon="भा",
        layout="centered",
    )
    inject_styles()

    language_label = st.selectbox("Language", list(LANGUAGE_OPTIONS), index=0)
    options = LANGUAGE_OPTIONS[language_label]
    language = str(options["code"])

    st.title(str(options["title"]))
    if language == "telugu":
        st.caption("Telugu predictions use the model trained on the public Alpaca and GPTeacher Telugu datasets. This is an experimental model; suggestions may be inaccurate.")
    else:
        st.caption("Baseline prototype: up to 5 suggestions when the training data supports the input. Vocabulary coverage is limited.")

    input_key = f"input_text_{language}"
    if input_key not in st.session_state:
        st.session_state[input_key] = str(options["default"])

    input_text = st.text_area(
        str(options["label"]),
        placeholder=str(options["placeholder"]),
        key=input_key,
    )

    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

    if predict_clicked:
        st.session_state["last_input_text"] = input_text
        st.session_state["last_language"] = language
        st.session_state["last_language_label"] = language_label

    if (
        "last_input_text" in st.session_state
        and st.session_state.get("last_language") == language
        and st.session_state["last_input_text"] == input_text
    ):
        render_predictions(st.session_state["last_input_text"], language, options)


if __name__ == "__main__":
    from telugu_app import main as current_main
    current_main()

