"""
predict.py
----------
Contains the full NLP preprocessing pipeline (identical to your research
notebook) and the prediction function that returns label, confidence,
probabilities, and top influential terms.
"""

import re
import string
import html as html_module
import numpy as np
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# ── NLTK setup (mirrors your notebook Cell 4) ─────────────────────
try:
    import nltk
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords as _sw
    STOP_WORDS = set(_sw.words("english"))
    from nltk.stem import WordNetLemmatizer
    _lem = WordNetLemmatizer()
    def _lemmatize(w: str) -> str:
        try:
            return _lem.lemmatize(w)
        except Exception:
            return w
    logger.info("NLTK WordNetLemmatizer loaded.")
except Exception as e:
    logger.warning(f"NLTK unavailable ({e}). Using fallback stopwords.")
    STOP_WORDS = {
        "i","me","my","we","our","you","your","he","him","his","she","her",
        "it","its","they","them","their","am","is","are","was","were","be",
        "been","have","has","had","do","does","did","a","an","the","and",
        "but","if","or","of","at","by","for","with","to","from","in","out",
        "on","not","can","will","just","should","now","s","t",
    }
    _lemmatize = lambda w: w

# ── Contraction map ────────────────────────────────────────────────
CONTRACTIONS: Dict[str, str] = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "i'm": "i am", "i've": "i have", "i'll": "i will",
    "i'd": "i would", "you're": "you are", "you've": "you have",
    "you'll": "you will", "you'd": "you would", "he's": "he is",
    "she's": "she is", "it's": "it is", "we're": "we are",
    "we've": "we have", "we'll": "we will", "they're": "they are",
    "they've": "they have", "they'll": "they will",
    "that's": "that is", "there's": "there is", "who's": "who is",
    "what's": "what is", "let's": "let us",
}

def _expand_contractions(text: str) -> str:
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
    return text

# ── Negation marking ───────────────────────────────────────────────
_NEG_RE = re.compile(
    r"\b(not|no|never|neither|nor|cannot|n\'t)\b\s+(\w+)",
    re.IGNORECASE,
)

def _mark_negations(text: str) -> str:
    return _NEG_RE.sub(lambda m: m.group(1) + " NOT_" + m.group(2), text)

# ── Reuters dateline (ISOT-specific) ─────────────────────────────
_DATELINE = re.compile(
    r"^[A-Z][A-Z\s,\.]+\s*\(Reuters\)\s*[\-\u2013\u2014]+\s*",
    re.MULTILINE,
)


# ── Main preprocessing function (mirrors notebook Cell 4) ─────────
def clean_text(
    text: str,
    remove_dateline: bool = True,
    expand_contrcts: bool = True,
    mark_neg: bool = True,
    remove_stopwords: bool = True,
    do_lemmatize: bool = True,
    min_len: int = 3,
) -> str:
    """
    Full 12-stage NLP preprocessing pipeline.
    Identical to the research notebook to ensure consistent feature extraction.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    if remove_dateline:
        text = _DATELINE.sub("", text)

    text = html_module.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", text)

    if expand_contrcts:
        text = _expand_contractions(text.lower())
    else:
        text = text.lower()

    if mark_neg:
        text = _mark_negations(text)

    text = re.sub(r"\b\d+\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z_\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) >= min_len]
    else:
        tokens = [t for t in tokens if len(t) >= min_len]

    if do_lemmatize:
        tokens = [_lemmatize(t) for t in tokens]

    return " ".join(tokens)


# ── Prediction function ────────────────────────────────────────────
def predict(raw_text: str, model_loader) -> Dict[str, Any]:
    """
    Run the full prediction pipeline on raw input text.

    Returns
    -------
    dict with keys:
        label           "Fake" or "Real"
        confidence      float 0–100 (max class probability %)
        prob_fake       float 0–100
        prob_real       float 0–100
        verdict         "FAKE" or "REAL" (upper, for display)
        clean_token_count  int
        top_fake_words  list of {word, score, direction}
        top_real_words  list of {word, score, direction}
        all_keywords    merged list for highlighting
        warning         str or None
    """
    # ── Input validation ──────────────────────────────────────────
    if not raw_text or not raw_text.strip():
        return {"error": "Input text is empty."}
    if len(raw_text.strip()) < 20:
        return {"error": "Input too short. Please paste a full headline or article."}

    # ── Preprocess ────────────────────────────────────────────────
    cleaned = clean_text(raw_text)
    token_count = len(cleaned.split())

    if token_count < 3:
        return {"error": "Not enough meaningful tokens after preprocessing. "
                         "Try a longer or more specific article."}

    # ── Vectorise ─────────────────────────────────────────────────
    X = model_loader.vectorizer.transform([cleaned])

    # ── Predict ───────────────────────────────────────────────────
    proba       = model_loader.model.predict_proba(X)[0]  # [P(fake), P(real)]
    pred_class  = int(np.argmax(proba))
    label       = model_loader.label_map[pred_class]
    confidence  = float(proba[pred_class]) * 100
    prob_fake   = float(proba[0]) * 100
    prob_real   = float(proba[1]) * 100

    # ── Top influential terms ─────────────────────────────────────
    vocabulary    = model_loader.vectorizer.vocabulary_
    diff_fake     = model_loader.diff_fake
    diff_real     = model_loader.diff_real
    feature_names = model_loader.feature_names

    tokens_in_vocab = [t for t in set(cleaned.split()) if t in vocabulary]

    # Sort by fake-direction score
    fake_scored = sorted(
        [(t, float(diff_fake[vocabulary[t]])) for t in tokens_in_vocab],
        key=lambda x: x[1], reverse=True
    )
    # Sort by real-direction score
    real_scored = sorted(
        [(t, float(diff_real[vocabulary[t]])) for t in tokens_in_vocab],
        key=lambda x: x[1], reverse=True
    )

    def fmt_words(pairs: List[Tuple[str, float]], direction: str, n: int = 8):
        return [
            {"word": w, "score": round(s, 4), "direction": direction}
            for w, s in pairs[:n] if s > 0
        ]

    top_fake_words = fmt_words(fake_scored, "fake")
    top_real_words = fmt_words(real_scored, "real")

    # All keywords for highlighting — top 12 by absolute score
    all_scored = sorted(
        [(t, float(diff_fake[vocabulary[t]])) for t in tokens_in_vocab],
        key=lambda x: abs(x[1]), reverse=True
    )
    all_keywords = [
        {
            "word": w,
            "score": round(s, 4),
            "direction": "fake" if s > 0 else "real",
        }
        for w, s in all_scored[:12]
    ]

    # ── Warning for low confidence ─────────────────────────────────
    warning = None
    if confidence < 65:
        warning = (
            "Low confidence prediction. The model is uncertain — "
            "consider reviewing this article manually."
        )

    return {
        "label"             : label,
        "verdict"           : label.upper(),
        "confidence"        : round(confidence, 2),
        "prob_fake"         : round(prob_fake, 2),
        "prob_real"         : round(prob_real, 2),
        "clean_token_count" : token_count,
        "top_fake_words"    : top_fake_words,
        "top_real_words"    : top_real_words,
        "all_keywords"      : all_keywords,
        "warning"           : warning,
    }
