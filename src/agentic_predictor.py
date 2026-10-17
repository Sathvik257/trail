from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ngram_predictor import NGramPredictor, build_ngram_model
from text_utils import LANGUAGE_RANGES, normalize_language, project_root, tokenize

try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment.
    torch = None


TELUGU_GENERIC_WORDS = {
    "కి",
    "కు",
    "లో",
    "తో",
    "ను",
    "గా",
    "ఈ",
    "ఆ",
    "ఒక",
    "కూడా",
    "మరియు",
    "అని",
    "అది",
    "ఇది",
    "ఉంది",
    "ఉన్నాయి",
    "పై",
    "లు",
    "ద్వారా",
    "కోసం",
    "నుంచి",
    "వరకు",
    "వంటి",
    "అన్ని",
    "చాలా",
    "లోనూ",
    "సమయంలో",
}
HINDI_GENERIC_WORDS = {
    "के",
    "की",
    "का",
    "को",
    "से",
    "में",
    "पर",
    "और",
    "यह",
    "वह",
    "एक",
    "भी",
    "है",
    "हैं",
    "था",
    "थी",
    "लिए",
    "द्वारा",
    "तक",
    "बहुत",
    "ने",
    "कि",
    "कर",
    "नहीं",
    "किया",
    "हो",
    "हुआ",
    "रहा",
    "रही",
    "रहे",
    "गया",
    "गई",
    "गए",
    "इस",
    "उस",
    "इन",
    "उन",
    "अपनी",
    "अपने",
    "लेकिन",
    "जब",
    "तब",
    "तो",
    "ही",
    "कहा",
    "करने",
    "हुए",
    "साथ",
    "वाले",
    "वाली",
    "होने",
    "लिए",
    "बाद",
    "पहले",
    "फिर",
}
ENGLISH_GENERIC_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "this",
    "that",
    "these",
    "those",
    "as",
    "at",
    "from",
    "not",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
}

TELUGU_PROJECT_CONTEXT_RULES = [
    ({"తెలుగు", "భాషను", "నేర్చుకోవాలి"}, ["అనుకుంటున్నాను", "రోజూ", "మాట్లాడాలి", "చదవాలి", "అభ్యసించాలి"]),
    ({"మొబైల్", "కీబోర్డ్"}, ["తదుపరి", "పదాన్ని", "సూచిస్తుంది", "చూపిస్తుంది", "త్వరగా"]),
    ({"కీబోర్డ్", "పదాలను"}, ["త్వరగా", "సూచిస్తుంది", "చూపిస్తుంది", "అంచనా", "ఇస్తుంది"]),
    ({"సర్వర్", "మోడల్"}, ["పంపిస్తుంది", "పారామీటర్లను", "సమీకరిస్తుంది", "మెరుగుపడుతుంది", "నవీకరిస్తుంది"]),
    ({"తెలంగాణలో", "కరోనా"}, ["కేసులు", "ప్రజలు", "ఆంక్షలు", "ఆసుపత్రులు", "పరీక్షలు"]),
    ({"కరోనా", "వైరస్"}, ["కేసులు", "ప్రజలు", "ఆంక్షలు", "ఆసుపత్రులు", "పరీక్షలు"]),
    ({"ఫెడరేటెడ్"}, ["లెర్నింగ్", "గోప్యతను", "డేటాను", "క్లయింట్లు", "శిక్షణ"]),
    ({"లెర్నింగ్"}, ["గోప్యతను", "డేటాను", "మోడల్", "క్లయింట్లు", "శిక్షణ"]),
    ({"సర్వర్"}, ["మోడల్", "పారామీటర్లను", "సమీకరిస్తుంది", "పంపిస్తుంది", "నవీకరిస్తుంది"]),
    ({"మోడల్"}, ["పంపిస్తుంది", "పారామీటర్లను", "సమీకరిస్తుంది", "మెరుగుపడుతుంది", "నవీకరిస్తుంది"]),
    ({"మొబైల్"}, ["కీబోర్డ్", "తదుపరి", "పదాన్ని", "సూచిస్తుంది", "చూపిస్తుంది"]),
    ({"కీబోర్డ్"}, ["తదుపరి", "పదాన్ని", "సూచిస్తుంది", "చూపిస్తుంది", "త్వరగా"]),
    ({"పదాలను"}, ["త్వరగా", "సూచిస్తుంది", "చూపిస్తుంది", "అంచనా", "ఇస్తుంది"]),
    ({"కరోనా"}, ["కేసులు", "ప్రజలు", "ఆంక్షలు", "ఆసుపత్రులు", "పరీక్షలు"]),
    ({"వైరస్"}, ["కేసులు", "ప్రజలు", "ఆంక్షలు", "ఆసుపత్రులు", "పరీక్షలు"]),
    ({"తెలంగాణలో"}, ["ప్రజలు", "కేసులు", "పరిస్థితి", "వర్షం", "అభివృద్ధి"]),
    ({"భారతదేశంలో"}, ["ప్రజలు", "భాషలు", "సాంకేతికత", "సేవలు", "అభివృద్ధి"]),
    ({"భాషను"}, ["నేర్చుకుంటున్నాను", "ప్రేమిస్తున్నాను", "మాట్లాడాలి", "చదవాలి", "అభ్యసించాలి"]),
    ({"నేర్చుకోవాలి"}, ["అనుకుంటున్నాను", "కాబట్టి", "రోజూ", "మాట్లాడాలి", "చదవాలి"]),
    ({"తెలుగు"}, ["భాషను", "నేర్చుకుంటున్నాను", "మాట్లాడతాను", "చదువుతున్నాను", "పదాలను"]),
]

HINDI_PROJECT_CONTEXT_RULES = [
    ({"मैं", "हिंदी"}, ["सीख", "भाषा", "बोलना", "पढ़ना", "लिखना"]),
    ({"मुझे", "हिंदी"}, ["भाषा", "सीखना", "बोलना", "पढ़ना", "लिखना"]),
    ({"हिंदी", "भाषा"}, ["सीखना", "सुंदर", "महत्वपूर्ण", "आसान", "समृद्ध"]),
    ({"मोबाइल", "कीबोर्ड"}, ["अगला", "शब्द", "सुझाता", "दिखाता", "जल्दी"]),
    ({"कीबोर्ड", "शब्दों"}, ["जल्दी", "सुझाता", "दिखाता", "अनुमान", "देता"]),
    ({"सर्वर", "मॉडल"}, ["भेजता", "पैरामीटर", "जोड़ता", "सुधारता", "अपडेट"]),
    ({"फेडरेटेड"}, ["लर्निंग", "गोपनीयता", "डेटा", "क्लाइंट", "प्रशिक्षण"]),
    ({"लर्निंग"}, ["गोपनीयता", "डेटा", "मॉडल", "क्लाइंट", "प्रशिक्षण"]),
    ({"कोरोना", "वायरस"}, ["मामले", "लोग", "अस्पताल", "जांच", "स्थिति"]),
    ({"भारत", "कोरोना"}, ["मामले", "लोग", "अस्पताल", "जांच", "स्थिति"]),
    ({"भारत", "में"}, ["लोग", "भाषाएं", "तकनीक", "सेवाएं", "विकास"]),
    ({"सर्वर"}, ["मॉडल", "पैरामीटर", "जोड़ता", "भेजता", "अपडेट"]),
    ({"मॉडल"}, ["भेजता", "पैरामीटर", "सुधारता", "अपडेट", "सीखता"]),
    ({"हिंदी"}, ["भाषा", "सीखना", "बोलना", "पढ़ना", "लिखना"]),
]

ENGLISH_PROJECT_CONTEXT_RULES = [
    ({"i", "english"}, ["learn", "speak", "read", "write", "practice"]),
    ({"learning", "english"}, ["language", "grammar", "words", "speaking", "practice"]),
    ({"english", "language"}, ["learning", "communication", "grammar", "vocabulary", "practice"]),
    ({"mobile", "keyboard"}, ["next", "word", "suggests", "predicts", "quickly"]),
    ({"keyboard", "words"}, ["quickly", "suggests", "predicts", "shows", "accurately"]),
    ({"server", "model"}, ["aggregates", "parameters", "updates", "improves", "learns"]),
    ({"federated"}, ["learning", "privacy", "data", "clients", "training"]),
    ({"learning"}, ["privacy", "data", "model", "clients", "training"]),
    ({"india", "in"}, ["people", "languages", "technology", "services", "development"]),
    ({"model"}, ["predicts", "learns", "improves", "updates", "suggests"]),
    ({"data"}, ["privacy", "security", "training", "clients", "model"]),
]

TELUGU_FALLBACK_WORDS = [
    "చేస్తుంది",
    "చూపిస్తుంది",
    "అవుతుంది",
    "ఉంటుంది",
    "పెరుగుతుంది",
    "సహాయపడుతుంది",
    "అవసరం",
    "ముఖ్యమైనది",
]
HINDI_FALLBACK_WORDS = [
    "करता",
    "दिखाता",
    "होता",
    "बढ़ता",
    "सीखता",
    "मदद",
    "जरूरी",
    "महत्वपूर्ण",
]
ENGLISH_FALLBACK_WORDS = [
    "helps",
    "improves",
    "predicts",
    "learns",
    "works",
    "supports",
    "suggests",
    "continues",
]

TELUGU_SENTENCE_RULES = [
    ({"నేను", "తెలుగు"}, [
        "భాషను నేర్చుకుంటున్నాను",
        "మాట్లాడటం అభ్యసిస్తున్నాను",
        "చదవడం మొదలుపెట్టాను",
        "పదాలను నేర్చుకుంటున్నాను",
        "భాషను ప్రేమిస్తున్నాను",
    ]),
    ({"తెలుగు", "భాష"}, [
        "చాలా అందమైన భాష",
        "నేర్చుకోవడం ఉపయోగకరం",
        "మాట్లాడటం సులభం",
        "పదసంపద సమృద్ధిగా ఉంటుంది",
        "మన సంస్కృతిని చూపిస్తుంది",
    ]),
    ({"మొబైల్", "కీబోర్డ్"}, [
        "తదుపరి పదాన్ని సూచిస్తుంది",
        "తెలుగు పదాలను త్వరగా చూపిస్తుంది",
        "వినియోగదారుకు వేగంగా టైప్ చేయడంలో సహాయపడుతుంది",
        "సందర్భాన్ని బట్టి పదాలను అంచనా వేస్తుంది",
        "ప్రైవసీని కాపాడుతూ పని చేస్తుంది",
    ]),
    ({"సర్వర్", "మోడల్"}, [
        "పారామీటర్లను సమీకరిస్తుంది",
        "నవీకరిస్తుంది",
        "క్లయింట్ల నుండి వచ్చిన అప్డేట్లను కలుపుతుంది",
        "గ్లోబల్ పనితీరును మెరుగుపరుస్తుంది",
        "కొత్త పదాలను మరింత బాగా నేర్చుకుంటుంది",
    ]),
    ({"ఫెడరేటెడ్", "లెర్నింగ్"}, [
        "గోప్యత కాపాడబడుతుంది",
        "డేటా క్లయింట్ల వద్దే ఉంటుంది",
        "మోడల్ సురక్షితంగా శిక్షణ పొందుతుంది",
        "క్లయింట్లు పారామీటర్లను మాత్రమే పంపుతాయి",
        "ముడి డేటా సర్వర్ కు పంపబడదు",
    ]),
    ({"ఫెడరేటెడ్"}, [
        "లెర్నింగ్ గోప్యతను కాపాడుతుంది",
        "లెర్నింగ్ లో డేటా క్లయింట్ల వద్దే ఉంటుంది",
        "లెర్నింగ్ ద్వారా మోడల్ సురక్షితంగా శిక్షణ పొందుతుంది",
        "లెర్నింగ్ మొబైల్ కీబోర్డ్ అప్లికేషన్లకు ఉపయోగపడుతుంది",
        "లెర్నింగ్ కేంద్ర సర్వర్ కు ముడి డేటాను పంపదు",
    ]),
    ({"లెర్నింగ్"}, [
        "గోప్యతను కాపాడుతూ మోడల్ ను శిక్షణ ఇస్తుంది",
        "డేటాను క్లయింట్ల వద్దే ఉంచుతుంది",
        "మోడల్ పనితీరును మెరుగుపరుస్తుంది",
        "సర్వర్ ద్వారా అప్డేట్లను సమీకరిస్తుంది",
        "భాషా సూచనలను మెరుగుపరుస్తుంది",
    ]),
    ({"భారతదేశంలో"}, [
        "ప్రజలు అనేక భాషలు మాట్లాడుతారు",
        "తెలుగు మరియు హిందీ విస్తృతంగా ఉపయోగిస్తారు",
        "సాంకేతిక సేవలు వేగంగా అభివృద్ధి చెందుతున్నాయి",
        "మొబైల్ వినియోగం పెరుగుతోంది",
        "కొత్త అప్లికేషన్లు ప్రజలకు సహాయపడుతున్నాయి",
    ]),
]

HINDI_SENTENCE_RULES = [
    ({"मैं", "हिंदी"}, [
        "भाषा सीख रहा हूँ",
        "बोलना सीख रहा हूँ",
        "पढ़ना शुरू कर रहा हूँ",
        "शब्दों का अभ्यास कर रहा हूँ",
        "भाषा को बेहतर समझ रहा हूँ",
    ]),
    ({"मुझे", "हिंदी"}, [
        "भाषा सीखनी है",
        "बोलना अच्छा लगता है",
        "पढ़ना और लिखना सीखना है",
        "शब्दों का अभ्यास करना है",
        "भाषा में सुधार करना है",
    ]),
    ({"हिंदी", "भाषा"}, [
        "सीखना आसान है",
        "बहुत सुंदर और समृद्ध है",
        "भारत की महत्वपूर्ण भाषा है",
        "लिखने और बोलने में उपयोगी है",
        "संस्कृति को अच्छी तरह दिखाती है",
    ]),
    ({"मोबाइल", "कीबोर्ड"}, [
        "अगला शब्द सुझाता है",
        "हिंदी शब्द जल्दी दिखाता है",
        "उपयोगकर्ता को तेज टाइपिंग में मदद करता है",
        "संदर्भ के अनुसार शब्दों का अनुमान लगाता है",
        "गोपनीयता बनाए रखते हुए काम करता है",
    ]),
    ({"सर्वर", "मॉडल"}, [
        "पैरामीटर को जोड़ता है",
        "अपडेट होता है",
        "क्लाइंट से आए अपडेट को मिलाता है",
        "ग्लोबल प्रदर्शन को बेहतर बनाता है",
        "नए शब्दों को और अच्छी तरह सीखता है",
    ]),
    ({"फेडरेटेड", "लर्निंग"}, [
        "गोपनीयता सुरक्षित रहती है",
        "डेटा क्लाइंट के पास रहता है",
        "मॉडल सुरक्षित रूप से प्रशिक्षित होता है",
        "क्लाइंट केवल पैरामीटर भेजते हैं",
        "कच्चा डेटा सर्वर को नहीं भेजा जाता है",
    ]),
    ({"फेडरेटेड"}, [
        "लर्निंग गोपनीयता को सुरक्षित रखती है",
        "लर्निंग में डेटा क्लाइंट के पास रहता है",
        "लर्निंग से मॉडल सुरक्षित रूप से प्रशिक्षित होता है",
        "लर्निंग मोबाइल कीबोर्ड एप्लिकेशन के लिए उपयोगी है",
        "लर्निंग कच्चा डेटा सर्वर को नहीं भेजती है",
    ]),
    ({"लर्निंग"}, [
        "गोपनीयता बचाते हुए मॉडल को प्रशिक्षित करती है",
        "डेटा को क्लाइंट के पास रखती है",
        "मॉडल की गुणवत्ता को बेहतर बनाती है",
        "सर्वर पर अपडेट को मिलाती है",
        "भाषा सुझावों को बेहतर बनाती है",
    ]),
    ({"भारत", "में"}, [
        "लोग कई भाषाएं बोलते हैं",
        "हिंदी व्यापक रूप से बोली जाती है",
        "तकनीक तेजी से विकसित हो रही है",
        "मोबाइल उपयोग बढ़ रहा है",
        "नई सेवाएं लोगों की मदद करती हैं",
    ]),
]

ENGLISH_SENTENCE_RULES = [
    ({"federated", "learning", "protects"}, [
        "user privacy",
        "raw data on local clients",
        "text data from central collection",
        "users by sharing only model parameters",
        "private typing data during training",
    ]),
    ({"i", "english"}, [
        "language every day",
        "speaking and writing",
        "words with regular practice",
        "grammar step by step",
        "communication skills",
    ]),
    ({"learning", "english"}, [
        "improves communication skills",
        "requires regular practice",
        "helps people speak clearly",
        "builds vocabulary and grammar",
        "makes writing more confident",
    ]),
    ({"english", "language"}, [
        "is useful for communication",
        "has a rich vocabulary",
        "is spoken around the world",
        "helps in education and technology",
        "can be learned with practice",
    ]),
    ({"mobile", "keyboard"}, [
        "suggests the next word",
        "predicts useful words quickly",
        "helps users type faster",
        "uses context for better suggestions",
        "can protect user privacy",
    ]),
    ({"server", "model"}, [
        "aggregates client parameters",
        "updates after every round",
        "combines local training results",
        "improves the global model",
        "learns better word patterns",
    ]),
    ({"federated", "learning"}, [
        "protects user privacy",
        "keeps data on local clients",
        "trains the model securely",
        "sends only model parameters",
        "improves predictions without sharing raw text",
    ]),
    ({"india", "in"}, [
        "people speak many languages",
        "technology is growing quickly",
        "mobile applications are widely used",
        "digital services support daily life",
        "language tools can help many users",
    ]),
]

TELUGU_SENTENCE_FALLBACKS = [
    "తదుపరి పదాలను స్పష్టంగా సూచిస్తుంది",
    "సందర్భాన్ని బట్టి వాక్యాన్ని పూర్తి చేస్తుంది",
    "మోడల్ సహాయంతో మరింత అర్థవంతంగా మారుతుంది",
    "వినియోగదారుకు వేగంగా టైప్ చేయడంలో సహాయపడుతుంది",
    "భాషా సూచనలను మరింత మెరుగుపరుస్తుంది",
]

HINDI_SENTENCE_FALLBACKS = [
    "अगले शब्दों को स्पष्ट रूप से सुझाता है",
    "संदर्भ के अनुसार वाक्य को पूरा करता है",
    "मॉडल की मदद से अधिक अर्थपूर्ण बनता है",
    "उपयोगकर्ता को तेज टाइपिंग में मदद करता है",
    "भाषा सुझावों को और बेहतर बनाता है",
]

ENGLISH_SENTENCE_FALLBACKS = [
    "gives a relevant next word suggestion",
    "continues the sentence using context",
    "helps users type faster",
    "improves language prediction quality",
    "supports clear sentence completion",
]

SENTENCE_RULES = {
    "telugu": TELUGU_SENTENCE_RULES,
    "hindi": HINDI_SENTENCE_RULES,
    "english": ENGLISH_SENTENCE_RULES,
}
SENTENCE_FALLBACKS = {
    "telugu": TELUGU_SENTENCE_FALLBACKS,
    "hindi": HINDI_SENTENCE_FALLBACKS,
    "english": ENGLISH_SENTENCE_FALLBACKS,
}
SENTENCE_ENDINGS = {
    "telugu": ".",
    "hindi": "।",
    "english": ".",
}


@dataclass(frozen=True)
class LanguageConfig:
    language: str
    data_dir: str
    raw_file: str
    training_file: str
    checkpoint_file: str
    context_rules: list[tuple[set[str], list[str]]]
    generic_words: set[str]
    fallback_words: list[str]


LANGUAGE_CONFIGS = {
    "telugu": LanguageConfig(
        language="telugu",
        data_dir="data_telugu_llm",
        raw_file="telugu_raw.txt",
        training_file="telugu_training.txt",
        checkpoint_file="telugu_llm_lstm.pt",
        context_rules=TELUGU_PROJECT_CONTEXT_RULES,
        generic_words=TELUGU_GENERIC_WORDS,
        fallback_words=TELUGU_FALLBACK_WORDS,
    ),
    "hindi": LanguageConfig(
        language="hindi",
        data_dir="data_hindi",
        raw_file="hindi_raw.txt",
        training_file="hindi_training.txt",
        checkpoint_file="hindi_lstm.pt",
        context_rules=HINDI_PROJECT_CONTEXT_RULES,
        generic_words=HINDI_GENERIC_WORDS,
        fallback_words=HINDI_FALLBACK_WORDS,
    ),
    "english": LanguageConfig(
        language="english",
        data_dir="data_english",
        raw_file="english_raw.txt",
        training_file="english_training.txt",
        checkpoint_file="english_lstm.pt",
        context_rules=ENGLISH_PROJECT_CONTEXT_RULES,
        generic_words=ENGLISH_GENERIC_WORDS,
        fallback_words=ENGLISH_FALLBACK_WORDS,
    ),
}


@dataclass(frozen=True)
class PredictionResult:
    input_text: str
    predictions: list[str]
    strategy: str
    confidence: float
    explanation: str


class AgenticPredictionService:
    """Routes prediction through small deterministic agents for cleaner Indic output."""

    def __init__(self, root: str | Path | None = None, language: str = "telugu") -> None:
        self.root = Path(root) if root else project_root()
        self.language = normalize_language(language)
        self.config = LANGUAGE_CONFIGS[self.language]

    @property
    def data_dir(self) -> Path:
        return self.root / self.config.data_dir

    @property
    def checkpoint_path(self) -> Path:
        return self.root / "outputs" / self.config.checkpoint_file

    def predict(self, text: str, top_k: int = 5, mode: str = "auto") -> PredictionResult:
        text = " ".join(tokenize(text, language=self.language))
        if not text:
            return PredictionResult("", [], "empty", 0.0, f"No {self.language} input was provided.")

        if mode == "auto":
            return self._auto_agent(text, top_k)

        if mode == "curated":
            return self._curated_context_agent(text, top_k)

        if mode == "ngram":
            corpus = self._large_context_agent(text, top_k)
            return corpus

        if mode == "lstm":
            return self._lstm_agent(text, top_k)

        raise ValueError(f"Unsupported prediction mode: {mode}")

    def predict_sentences(
        self,
        text: str,
        count: int = 5,
        max_new_words: int = 7,
        mode: str = "auto",
    ) -> PredictionResult:
        text = " ".join(tokenize(text, language=self.language))
        if not text:
            return PredictionResult("", [], "empty", 0.0, f"No {self.language} input was provided.")

        # Telugu uses the new corpus and checkpoint; other languages use corpus matches.
        sentences: list[str] = []
        candidates: list[str] = []
        for agent in self._sentence_agents():
            self._add_unique(candidates, agent(text, count).predictions, count)
        for word in candidates:
            self._add_sentence(sentences, self._generate_sentence(text, word, max_new_words), count)
        source = "corpus-lstm" if self.language == "telugu" else "corpus"
        return self._sentence_result(text, sentences, 0.0, source)

    def _sentence_agents(self):
        if self.language == "telugu":
            return (self._large_context_agent, self._lstm_agent)
        return (self._curated_context_agent, self._large_context_agent)

    def _auto_agent(self, text: str, top_k: int) -> PredictionResult:
        if self.language in {"hindi", "english"}:
            agents = [
                self._project_context_agent,
                self._curated_context_agent,
                self._large_context_agent,
                self._lstm_agent,
            ]
        else:
            agents = [
                self._lstm_agent,
                self._large_context_agent,
            ]

        predictions: list[str] = []
        active_sources: list[str] = []
        confidence = 0.0
        for agent in agents:
            result = agent(text, top_k)
            before_count = len(predictions)
            self._add_unique(predictions, result.predictions, top_k)
            if len(predictions) > before_count:
                active_sources.append(result.strategy)
            if confidence == 0.0 and result.predictions:
                confidence = result.confidence
            if len(predictions) == top_k:
                break

        active = ", ".join(dict.fromkeys(active_sources)) or "none"
        return PredictionResult(
            input_text=text,
            predictions=predictions,
            strategy="agentic",
            confidence=confidence,
            explanation=f"Combined clean {self.language} suggestions from: {active}.",
        )

    def _curated_context_agent(self, text: str, top_k: int) -> PredictionResult:
        predictor = load_curated_predictor(self.root, self.language)
        predictions = self._quality_filter(predictor.predict_contextual(text, top_k=top_k), top_k)
        return PredictionResult(
            input_text=text,
            predictions=predictions,
            strategy="curated",
            confidence=0.95 if predictions else 0.0,
            explanation=f"Matched clear project-specific {self.language} examples.",
        )

    def _project_context_agent(self, text: str, top_k: int) -> PredictionResult:
        tokens = set(tokenize(text, language=self.language))
        predictions: list[str] = []
        matched_rules = [
            (len(required_tokens), words)
            for required_tokens, words in self.config.context_rules
            if required_tokens.issubset(tokens)
        ]
        matched_rules.sort(key=lambda item: item[0], reverse=True)

        for _, words in matched_rules:
            self._add_unique(predictions, words, top_k, avoid_words=tokens)
            if len(predictions) == top_k:
                break

        predictions = self._quality_filter(predictions, top_k, drop_generic=True)
        return PredictionResult(
            input_text=text,
            predictions=predictions,
            strategy="project",
            confidence=0.9 if predictions else 0.0,
            explanation=f"Used project-specific {self.language} context hints before statistical models.",
        )

    def _large_context_agent(self, text: str, top_k: int) -> PredictionResult:
        predictor = load_large_predictor(self.root, self.language)
        predictions = self._quality_filter(
            predictor.predict_contextual(text, top_k=top_k * 3),
            top_k,
            drop_generic=True,
        )
        return PredictionResult(
            input_text=text,
            predictions=predictions,
            strategy="ngram",
            confidence=0.82 if predictions else 0.0,
            explanation=f"Matched the larger {self.language} corpus context.",
        )

    def _large_fallback_agent(self, text: str, top_k: int) -> PredictionResult:
        predictor = load_large_predictor(self.root, self.language)
        predictions = self._quality_filter(
            predictor.predict(text, top_k=top_k * 3),
            top_k,
            drop_generic=True,
        )
        return PredictionResult(
            input_text=text,
            predictions=predictions,
            strategy="ngram",
            confidence=0.45 if predictions else 0.0,
            explanation="Used corpus frequency fallback because no exact context matched.",
        )

    def _lstm_agent(self, text: str, top_k: int) -> PredictionResult:
        bundle = load_lstm_bundle(self.checkpoint_path)
        if bundle is None:
            return PredictionResult(text, [], "lstm", 0.0, "LSTM checkpoint is not available.")

        model, word_to_idx, idx_to_word, seq_length = bundle
        pad_id = word_to_idx.get("<pad>", 0)
        unk_id = word_to_idx.get("<unk>", 1)
        ids = [word_to_idx.get(token, unk_id) for token in tokenize(text, language=self.language)][-seq_length:]
        if not ids or ids[-1] == unk_id:
            return PredictionResult(text, [], "unsupported", 0.0, "The final word is outside the trained vocabulary. Try a longer phrase with a known final word.")

        if len(ids) < seq_length:
            ids = [pad_id] * (seq_length - len(ids)) + ids

        with torch.no_grad():
            input_ids = torch.tensor([ids], dtype=torch.long)
            logits = model(input_ids)[0]
            probabilities = torch.softmax(logits, dim=0)
            candidate_count = min(len(word_to_idx), max(top_k * 10, top_k))
            values, indices = torch.topk(probabilities, k=candidate_count)

        raw_predictions = [idx_to_word[int(idx)] for idx in indices.tolist()]
        predictions = self._quality_filter(raw_predictions, top_k, drop_generic=True)
        confidence = float(values[0].item()) if len(values) else 0.0
        return PredictionResult(
            input_text=text,
            predictions=predictions,
            strategy="lstm",
            confidence=confidence,
            explanation="Used the trained next-word LSTM model.",
        )

    def _template_sentence_candidates(self, text: str) -> list[str]:
        tokens = set(tokenize(text, language=self.language))
        matched_rules = [
            (len(required_tokens), suffixes)
            for required_tokens, suffixes in SENTENCE_RULES[self.language]
            if required_tokens.issubset(tokens)
        ]
        matched_rules.sort(key=lambda item: item[0], reverse=True)

        sentences: list[str] = []
        for _, suffixes in matched_rules:
            for suffix in suffixes:
                self._add_sentence(sentences, self._compose_sentence(text, suffix), top_k=5)
        return sentences

    def _generate_sentence(self, text: str, first_word: str, max_new_words: int) -> str:
        base_tokens = tokenize(text, language=self.language)
        generated_tokens = [first_word]
        used_words = set(base_tokens)

        for _ in range(max(0, max_new_words - 1)):
            context = " ".join([*base_tokens, *generated_tokens])
            candidates: list[str] = []
            for agent in self._sentence_agents():
                result = agent(context, top_k=10)
                candidates.extend(result.predictions)

            next_word = self._first_sentence_word(candidates, used_words | set(generated_tokens))
            if not next_word:
                break
            generated_tokens.append(next_word)
            if self._looks_sentence_terminal(next_word):
                break

        return self._compose_sentence(text, " ".join(generated_tokens))

    def _safe_sentence_from_word(self, text: str, word: str) -> str:
        suffixes = {
            "hindi": {
                "कोरोना": "कोरोना के मामले चर्चा में हैं",
                "हिंदी": "हिंदी भाषा व्यापक रूप से बोली जाती है",
                "लॉन्च": "लॉन्च से जुड़ी जानकारी मिलती है",
                "सबसे": "सबसे महत्वपूर्ण जानकारी सामने आती है",
                "डेटा": "डेटा सुरक्षित रहता है",
                "मॉडल": "मॉडल बेहतर परिणाम देता है",
                "क्लाइंट": "क्लाइंट स्थानीय रूप से प्रशिक्षण करते हैं",
                "प्रशिक्षण": "प्रशिक्षण से गुणवत्ता बेहतर होती है",
                "गोपनीयता": "गोपनीयता सुरक्षित रहती है",
                "लोग": "लोग कई भाषाएं बोलते हैं",
                "भाषाएं": "भाषाएं संस्कृति को जोड़ती हैं",
                "तकनीक": "तकनीक तेजी से विकसित हो रही है",
                "सेवाएं": "सेवाएं लोगों की मदद करती हैं",
                "विकास": "विकास लगातार आगे बढ़ रहा है",
            },
            "telugu": {
                "కరోనా": "కరోనా కేసుల గురించి సమాచారం లభిస్తుంది",
                "అతిపెద్ద": "అతిపెద్ద మార్కెట్ గా గుర్తింపు పొందింది",
                "అత్యంత": "అత్యంత ముఖ్యమైన సమాచారం అందిస్తుంది",
                "అత్యధిక": "అత్యధిక వినియోగం కనిపిస్తుంది",
                "డేటా": "డేటా సురక్షితంగా ఉంటుంది",
                "మోడల్": "మోడల్ మెరుగైన ఫలితాలు ఇస్తుంది",
                "క్లయింట్లు": "క్లయింట్లు స్థానికంగా శిక్షణ పొందుతాయి",
                "శిక్షణ": "శిక్షణ ద్వారా నాణ్యత మెరుగుపడుతుంది",
                "గోప్యతను": "గోప్యతను కాపాడుతుంది",
                "ప్రజలు": "ప్రజలు అనేక భాషలు మాట్లాడుతారు",
                "భాషలు": "భాషలు సంస్కృతిని కలుపుతాయి",
                "సాంకేతికత": "సాంకేతికత వేగంగా అభివృద్ధి చెందుతోంది",
                "సేవలు": "సేవలు ప్రజలకు సహాయపడుతున్నాయి",
                "అభివృద్ధి": "అభివృద్ధి నిరంతరం ముందుకు సాగుతోంది",
            },
            "english": {
                "corona": "corona cases are discussed carefully",
                "english": "english language is useful for communication",
                "learning": "learning improves language skills",
                "privacy": "privacy remains protected",
                "data": "data stays secure",
                "model": "model gives better predictions",
                "clients": "clients train locally",
                "training": "training improves prediction quality",
                "people": "people speak many languages",
                "languages": "languages connect people and culture",
                "technology": "technology is growing quickly",
                "services": "services help people every day",
                "development": "development continues to move forward",
                "word": "word prediction becomes more accurate",
                "suggests": "suggests useful next words",
                "predicts": "predicts relevant text from context",
            },
        }
        fallback = {
            "hindi": f"{word} से संबंधित सही सुझाव मिलता है",
            "telugu": f"{word} కు సంబంధించిన సరైన సూచన లభిస్తుంది",
            "english": f"{word} gives a relevant continuation",
        }
        suffix = suffixes[self.language].get(word, fallback[self.language])
        return self._compose_sentence(text, suffix)

    def _first_sentence_word(self, words: list[str], used_words: set[str]) -> str:
        for word in words:
            if word in used_words:
                continue
            if word in self.config.generic_words and len(used_words) > 2:
                continue
            if self._has_language_char(word):
                return word
        return ""

    def _compose_sentence(self, text: str, suffix: str) -> str:
        base_tokens = tokenize(text, language=self.language)
        suffix_tokens = tokenize(suffix, language=self.language)
        if not suffix_tokens:
            tokens = base_tokens
        elif suffix_tokens[: len(base_tokens)] == base_tokens:
            tokens = suffix_tokens
        else:
            tokens = [*base_tokens, *self._dedupe_boundary(base_tokens, suffix_tokens)]

        sentence = " ".join(tokens).strip()
        return sentence.rstrip(" .।") + SENTENCE_ENDINGS[self.language]

    @staticmethod
    def _dedupe_boundary(base_tokens: list[str], suffix_tokens: list[str]) -> list[str]:
        if not base_tokens or not suffix_tokens:
            return suffix_tokens
        max_overlap = min(len(base_tokens), len(suffix_tokens))
        for size in range(max_overlap, 0, -1):
            if base_tokens[-size:] == suffix_tokens[:size]:
                return suffix_tokens[size:]
        return suffix_tokens

    def _looks_sentence_terminal(self, word: str) -> bool:
        if self.language == "hindi":
            return word.endswith(("है", "हैं", "हूँ", "होता", "करता", "देता", "रखता"))
        if self.language == "english":
            return word.endswith(("s", "ed", "ing"))
        return word.endswith(("ది", "డు", "ను", "తుంది", "తాడు", "తాను", "స్తుంది", "పడుతుంది"))

    def _sentence_result(
        self,
        text: str,
        sentences: list[str],
        confidence: float,
        source: str,
    ) -> PredictionResult:
        return PredictionResult(
            input_text=text,
            predictions=sentences,
            strategy=f"sentence-{source}",
            confidence=confidence,
            explanation=f"Generated complete {self.language} sentence continuations.",
        )

    def _quality_filter(self, words: list[str], top_k: int, drop_generic: bool = False) -> list[str]:
        clean: list[str] = []
        for word in words:
            if word in {"<pad>", "<unk>", "</s>"}:
                continue
            if drop_generic and word in self.config.generic_words:
                continue
            if not self._has_language_char(word):
                continue
            if word not in clean:
                clean.append(word)
            if len(clean) == top_k:
                break
        return clean

    def _has_language_char(self, word: str) -> bool:
        start, end = LANGUAGE_RANGES[self.language]
        return any(start <= char <= end for char in word)

    @staticmethod
    def _add_unique(
        target: list[str],
        words: list[str],
        top_k: int,
        avoid_words: set[str] | None = None,
    ) -> None:
        avoid_words = avoid_words or set()
        for word in words:
            if word in avoid_words:
                continue
            if word not in target:
                target.append(word)
            if len(target) == top_k:
                return

    @staticmethod
    def _add_sentence(target: list[str], sentence: str, top_k: int) -> None:
        sentence = " ".join(sentence.split())
        if sentence and sentence not in target:
            target.append(sentence)
        if len(target) > top_k:
            del target[top_k:]


@lru_cache(maxsize=8)
def load_curated_predictor(root: str | Path, language: str) -> NGramPredictor:
    if normalize_language(language) == "telugu":
        return load_large_predictor(root, language)
    root = Path(root)
    config = LANGUAGE_CONFIGS[normalize_language(language)]
    data_dir = root / config.data_dir
    model_path = data_dir / "curated_ngram_model.pkl"
    if model_path.exists():
        return NGramPredictor.load(model_path)
    return build_ngram_model(
        data_dir / config.raw_file,
        model_path,
        max_order=5,
        max_lines=0,
        language=config.language,
    )


@lru_cache(maxsize=8)
def load_large_predictor(root: str | Path, language: str) -> NGramPredictor:
    root = Path(root)
    config = LANGUAGE_CONFIGS[normalize_language(language)]
    data_dir = root / config.data_dir
    model_path = data_dir / "ngram_model.pkl"
    if model_path.exists():
        return NGramPredictor.load(model_path)
    raw_path = data_dir / config.training_file
    if not raw_path.exists():
        raw_path = data_dir / config.raw_file
    return build_ngram_model(raw_path, model_path, max_order=5, max_lines=15000, language=config.language)


def load_lstm_bundle(model_path: str | Path):
    if torch is None:
        return None

    model_path = Path(model_path)
    if not model_path.exists():
        return None

    revision = model_path.stat()
    return _load_lstm_bundle_cached(str(model_path.resolve()), revision.st_mtime_ns, revision.st_size)


@lru_cache(maxsize=4)
def _load_lstm_bundle_cached(model_path: str, modified_ns: int, size: int):
    # Include the file revision so a running frontend sees newly saved checkpoints.

    from model import TeluguLSTM

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    word_to_idx = checkpoint["word_to_idx"]
    idx_to_word = checkpoint["idx_to_word"]
    pad_id = word_to_idx.get("<pad>", 0)

    model = TeluguLSTM(
        vocab_size=len(word_to_idx),
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        pad_idx=pad_id,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, word_to_idx, idx_to_word, checkpoint["seq_length"]
