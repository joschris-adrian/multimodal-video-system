import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.decomposition import LatentDirichletAllocation

# stopwords (no NLTK needed) 

STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "and", "or",
    "but", "i", "we", "you", "he", "she", "they", "was", "were", "be",
    "been", "have", "has", "that", "this", "of", "for", "with", "are",
    "just", "so", "my", "your", "its", "our", "their", "do", "did",
    "not", "no", "up", "out", "by", "from", "about", "as", "if", "than"
}

SENTIMENT_POS = {
    "good", "great", "love", "beautiful", "amazing", "happy", "excited",
    "wonderful", "fantastic", "enjoy", "fun", "best", "excellent", "perfect"
}
SENTIMENT_NEG = {
    "bad", "hate", "terrible", "awful", "sad", "angry", "worst",
    "horrible", "fail", "wrong", "problem", "difficult", "never", "nothing"
}


# text cleaning 

def clean(text: str) -> list[str]:
    """Lowercase, remove punctuation, remove stopwords."""
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS]


# TF-IDF 

def tfidf_keywords(texts: list[str], top_n: int = 5) -> list[list[str]]:
    """
    Compute TF-IDF across a corpus of transcripts.
    Returns top_n keywords per document.

    Usage:
        texts = [video1_transcript, video2_transcript, ...]
        keywords_per_video = tfidf_keywords(texts)
    """
    vectorizer = TfidfVectorizer(
        stop_words  = "english",
        max_features= 500,
        ngram_range = (1, 2),   # unigrams + bigrams
    )

    try:
        X      = vectorizer.fit_transform(texts)
        terms  = vectorizer.get_feature_names_out()
        result = []

        for i in range(X.shape[0]):
            row     = X[i].toarray().flatten()
            indices = row.argsort()[::-1][:top_n]
            result.append([terms[j] for j in indices if row[j] > 0])

        return result
    except Exception:
        return [[] for _ in texts]


def tfidf_vector(text: str, vocabulary: list[str] | None = None) -> np.ndarray:
    """
    TF-IDF embedding for a single transcript.
    Returns a 1D numpy array.
    """
    params = {"stop_words": "english", "max_features": 300}
    if vocabulary:
        params["vocabulary"] = vocabulary

    vec = TfidfVectorizer(**params)
    try:
        return vec.fit_transform([text]).toarray().flatten()
    except Exception:
        return np.zeros(300)


# Word2Vec 

_w2v_model = None

def train_word2vec(texts: list[str], vector_size: int = 64) -> object:
    """
    Train Word2Vec on a corpus of transcripts.
    Call once across all videos, then reuse the model.
    """
    global _w2v_model
    sentences   = [clean(t) for t in texts if t.strip()]
    _w2v_model  = Word2Vec(
        sentences,
        vector_size = vector_size,
        window      = 5,
        min_count   = 1,
        workers     = 2,
        epochs      = 10,
    )
    return _w2v_model


def word2vec_embedding(text: str, vector_size: int = 64) -> np.ndarray:
    """
    Mean-pool Word2Vec vectors for all words in a transcript.
    Returns a 1D numpy array of shape (vector_size,).
    """
    global _w2v_model
    if _w2v_model is None:
        return np.zeros(vector_size)

    words  = clean(text)
    vecs   = [
        _w2v_model.wv[w]
        for w in words
        if w in _w2v_model.wv
    ]
    return np.mean(vecs, axis=0) if vecs else np.zeros(vector_size)


# BERT embeddings 

_bert_model     = None
_bert_tokenizer = None

def _load_bert():
    global _bert_model, _bert_tokenizer
    if _bert_model is None:

        name             = "sentence-transformers/all-MiniLM-L6-v2"  # 80MB, fast
        _bert_tokenizer  = AutoTokenizer.from_pretrained(name)
        _bert_model      = AutoModel.from_pretrained(name)
        _bert_model.eval()
    return _bert_tokenizer, _bert_model


def bert_embedding(text: str) -> np.ndarray:
    """
    Sentence embedding via MiniLM (lightweight BERT variant).
    Returns a 1D numpy array of shape (384,).
    Falls back to zeros if text is empty.
    """
    if not text.strip():
        return np.zeros(384)

    tokenizer, model = _load_bert()

    inputs = tokenizer(
        text,
        return_tensors   = "pt",
        truncation       = True,
        max_length       = 128,
        padding          = True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # mean pool over token embeddings
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding


def embedding_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embeddings."""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Sentiment 

def sentiment(text: str) -> dict:
    """Rule-based sentiment. Returns label + score in [-1, 1]."""
    words = set(text.lower().split())
    pos   = len(words & SENTIMENT_POS)
    neg   = len(words & SENTIMENT_NEG)
    total = pos + neg

    if total == 0:
        return {"label": "neutral", "score": 0.0}

    score = (pos - neg) / total
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    return {"label": label, "score": round(score, 2)}


# LDA Topic Modelling 

_lda_model      = None
_lda_vocabulary = None

def train_lda(texts: list[str], n_topics: int = 5) -> object:
    """
    Fit LDA on a corpus of transcripts.
    Call once across all videos, then use get_topics() per video.
    """
    global _lda_model, _lda_vocabulary

    vectorizer    = CountVectorizer(stop_words="english", max_features=300)
    X             = vectorizer.fit_transform(texts)
    _lda_vocabulary = vectorizer

    _lda_model = LatentDirichletAllocation(
        n_components  = n_topics,
        random_state  = 42,
        max_iter      = 20,
    )
    _lda_model.fit(X)
    return _lda_model


def get_topics(text: str, top_words: int = 5) -> list[dict]:
    """
    Get topic distribution for a single transcript.
    Returns list of {topic_id, weight, words} sorted by weight.
    """
    global _lda_model, _lda_vocabulary

    if _lda_model is None or not text.strip():
        return []

    X       = _lda_vocabulary.transform([text])
    dist    = _lda_model.transform(X)[0]           # topic distribution
    terms   = _lda_vocabulary.get_feature_names_out()
    result  = []

    for topic_id, weight in enumerate(dist):
        top_indices = _lda_model.components_[topic_id].argsort()[::-1][:top_words]
        result.append({
            "topic_id": topic_id,
            "weight":   round(float(weight), 3),
            "words":    [terms[i] for i in top_indices],
        })

    return sorted(result, key=lambda x: -x["weight"])


def dominant_topic(text: str) -> dict | None:
    """Return the single highest-weight topic for a transcript."""
    topics = get_topics(text)
    return topics[0] if topics else None


# Full analysis (single transcript) 

def analyze_transcript(text: str, use_bert: bool = False) -> dict:
    """
    Full NLP analysis for a single transcript.

    Returns:
        keywords        - top content words (simple frequency)
        tfidf_vector    - TF-IDF embedding (300-dim)
        bert_embedding  - MiniLM sentence embedding (384-dim, optional)
        sentiment       - label + score
        topic           - dominant LDA topic (if model trained)
        word_count
        sentence_count
        avg_sentence_length
    """
    if not text.strip():
        return {
            "keywords":           [],
            "tfidf_vector":       np.zeros(300),
            "bert_embedding":     np.zeros(384) if use_bert else None,
            "sentiment":          {"label": "neutral", "score": 0.0},
            "topic":              None,
            "word_count":         0,
            "sentence_count":     0,
            "avg_sentence_length": 0.0,
        }

    words     = text.split()
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    keywords  = [w for w, _ in Counter(clean(text)).most_common(5)]

    return {
        "keywords":            keywords,
        "tfidf_vector":        tfidf_vector(text),
        "bert_embedding":      bert_embedding(text) if use_bert else None,
        "sentiment":           sentiment(text),
        "topic":               dominant_topic(text),
        "word_count":          len(words),
        "sentence_count":      len(sentences),
        "avg_sentence_length": round(len(words) / max(len(sentences), 1), 1),
    }