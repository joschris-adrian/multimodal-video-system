# tests/test_nlp_analysis.py

import pytest
import numpy as np


# clean 

def test_clean_removes_stopwords():
    from src.audio.nlp_analysis import clean
    result = clean("the person is walking in the street")
    assert "the" not in result
    assert "is"  not in result
    assert "in"  not in result


def test_clean_removes_short_words():
    from src.audio.nlp_analysis import clean
    result = clean("a be do it go")
    assert result == []


def test_clean_lowercases():
    from src.audio.nlp_analysis import clean
    result = clean("WALKING Running SURFING")
    assert "walking" in result
    assert "running" in result


def test_clean_returns_list():
    from src.audio.nlp_analysis import clean
    assert isinstance(clean("some sample text here"), list)


def test_clean_empty_string():
    from src.audio.nlp_analysis import clean
    assert clean("") == []


# sentiment 

def test_sentiment_positive():
    from src.audio.nlp_analysis import sentiment
    result = sentiment("This is great amazing wonderful fun")
    assert result["label"] == "positive"
    assert result["score"] > 0


def test_sentiment_negative():
    from src.audio.nlp_analysis import sentiment
    result = sentiment("This is terrible awful horrible bad")
    assert result["label"] == "negative"
    assert result["score"] < 0


def test_sentiment_neutral():
    from src.audio.nlp_analysis import sentiment
    result = sentiment("The person walked across the room")
    assert result["label"] == "neutral"
    assert result["score"] == 0.0


def test_sentiment_empty():
    from src.audio.nlp_analysis import sentiment
    result = sentiment("")
    assert result["label"] == "neutral"
    assert result["score"] == 0.0


def test_sentiment_returns_score_in_range():
    from src.audio.nlp_analysis import sentiment
    result = sentiment("good great love amazing bad hate")
    assert -1.0 <= result["score"] <= 1.0


def test_sentiment_returns_expected_keys():
    from src.audio.nlp_analysis import sentiment
    result = sentiment("some text")
    assert "label" in result
    assert "score" in result


# tfidf_vector 

def test_tfidf_vector_returns_numpy_array():
    from src.audio.nlp_analysis import tfidf_vector
    result = tfidf_vector("person walking down the street with a car")
    assert isinstance(result, np.ndarray)


def test_tfidf_vector_nonzero_for_real_text():
    from src.audio.nlp_analysis import tfidf_vector
    result = tfidf_vector("person walking down the busy urban street")
    assert result.sum() > 0


def test_tfidf_vector_zeros_for_empty():
    from src.audio.nlp_analysis import tfidf_vector
    result = tfidf_vector("")
    assert result.sum() == 0.0


def test_tfidf_vector_consistent_dim():
    from src.audio.nlp_analysis import tfidf_vector
    r1 = tfidf_vector("person walking in the street")
    r2 = tfidf_vector("surfing on the beach with waves")
    assert r1.shape == r2.shape


# tfidf_keywords 

def test_tfidf_keywords_returns_list_of_lists():
    from src.audio.nlp_analysis import tfidf_keywords
    texts  = ["person walking street car", "surfing beach waves board"]
    result = tfidf_keywords(texts, top_n=3)
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], list)


def test_tfidf_keywords_top_n_respected():
    from src.audio.nlp_analysis import tfidf_keywords
    texts  = [
        "person walking down the busy street with cars and buses",
        "surfing on the beautiful sunny beach with waves and board",
    ]
    result = tfidf_keywords(texts, top_n=3)
    for kws in result:
        assert len(kws) <= 3


def test_tfidf_keywords_single_text():
    from src.audio.nlp_analysis import tfidf_keywords
    result = tfidf_keywords(["person walking street"], top_n=3)
    assert isinstance(result, list)
    assert len(result) == 1


def test_tfidf_keywords_empty_texts():
    from src.audio.nlp_analysis import tfidf_keywords
    result = tfidf_keywords(["", ""], top_n=3)
    assert isinstance(result, list)


# word2vec 

def test_word2vec_embedding_before_training_returns_zeros():
    import importlib
    import src.audio.nlp_analysis as m
    m._w2v_model = None   # reset global
    result = m.word2vec_embedding("person walking street")
    assert isinstance(result, np.ndarray)
    assert result.sum() == 0.0


def test_train_word2vec_returns_model():
    from src.audio.nlp_analysis import train_word2vec
    texts  = [
        "person walking down the street",
        "surfing on the beach with waves",
        "playing soccer in the sports field",
    ]
    model = train_word2vec(texts, vector_size=16)
    assert model is not None


def test_word2vec_embedding_after_training():
    from src.audio.nlp_analysis import train_word2vec, word2vec_embedding
    texts = [
        "person walking street car road",
        "surfing beach waves board water",
        "dancing music indoor floor stage",
    ]
    train_word2vec(texts, vector_size=16)
    result = word2vec_embedding("person walking street", vector_size=16)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 16


def test_word2vec_empty_text_after_training():
    from src.audio.nlp_analysis import train_word2vec, word2vec_embedding
    texts = ["person walking street", "surfing beach waves"]
    train_word2vec(texts, vector_size=16)
    result = word2vec_embedding("", vector_size=16)
    assert result.sum() == 0.0


# bert embedding 

def test_bert_embedding_returns_numpy_array():
    from src.audio.nlp_analysis import bert_embedding
    result = bert_embedding("A person walking down the street")
    assert isinstance(result, np.ndarray)


def test_bert_embedding_dim():
    from src.audio.nlp_analysis import bert_embedding
    result = bert_embedding("A person walking down the street")
    assert result.shape[0] == 384


def test_bert_embedding_empty_text():
    from src.audio.nlp_analysis import bert_embedding
    result = bert_embedding("")
    assert result.sum() == 0.0
    assert result.shape[0] == 384


def test_bert_embedding_nonzero_for_real_text():
    from src.audio.nlp_analysis import bert_embedding
    result = bert_embedding("A person is surfing on the beach")
    assert result.sum() != 0.0


def test_bert_embedding_different_texts_differ():
    from src.audio.nlp_analysis import bert_embedding
    a = bert_embedding("A person walking in the street")
    b = bert_embedding("A dog running on the beach")
    assert not np.allclose(a, b)


def test_embedding_similarity_identical():
    from src.audio.nlp_analysis import embedding_similarity
    v = np.array([1.0, 0.0, 0.0])
    assert embedding_similarity(v, v) == pytest.approx(1.0)


def test_embedding_similarity_orthogonal():
    from src.audio.nlp_analysis import embedding_similarity
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert embedding_similarity(a, b) == pytest.approx(0.0)


def test_embedding_similarity_zero_vector():
    from src.audio.nlp_analysis import embedding_similarity
    a = np.array([1.0, 0.0])
    b = np.zeros(2)
    assert embedding_similarity(a, b) == 0.0


# lda 

def test_train_lda_returns_model():
    from src.audio.nlp_analysis import train_lda
    texts = [
        "person walking street car road urban",
        "surfing beach waves board water ocean",
        "dancing music indoor floor stage performance",
        "cooking kitchen food eating restaurant meal",
        "playing soccer field sport ball team",
    ]
    model = train_lda(texts, n_topics=3)
    assert model is not None


def test_get_topics_returns_list():
    from src.audio.nlp_analysis import train_lda, get_topics
    texts = [
        "person walking street car road",
        "surfing beach waves board water",
        "dancing music indoor floor stage",
        "cooking kitchen food eating meal",
        "playing soccer field sport ball",
    ]
    train_lda(texts, n_topics=3)
    result = get_topics("person walking down the street")
    assert isinstance(result, list)
    assert len(result) == 3


def test_get_topics_weight_sum_approx_one():
    from src.audio.nlp_analysis import train_lda, get_topics
    texts = [
        "person walking street car road",
        "surfing beach waves board water",
        "dancing music indoor floor stage",
        "cooking kitchen food eating meal",
        "playing soccer field sport ball",
    ]
    train_lda(texts, n_topics=3)
    result = get_topics("person walking down the street")
    total  = sum(t["weight"] for t in result)
    assert total == pytest.approx(1.0, abs=0.01)


def test_get_topics_sorted_by_weight():
    from src.audio.nlp_analysis import train_lda, get_topics
    texts = [
        "person walking street car road",
        "surfing beach waves board water",
        "dancing music indoor floor stage",
        "cooking kitchen food eating meal",
        "playing soccer field sport ball",
    ]
    train_lda(texts, n_topics=3)
    result  = get_topics("person walking down the street")
    weights = [t["weight"] for t in result]
    assert weights == sorted(weights, reverse=True)


def test_get_topics_topic_keys():
    from src.audio.nlp_analysis import train_lda, get_topics
    texts = [
        "person walking street car road urban traffic",
        "surfing beach waves board water ocean tide",
        "dancing music indoor floor stage performance art",
        "cooking kitchen food eating restaurant meal chef",
        "playing soccer field sport ball team match",
    ]
    train_lda(texts, n_topics=3)
    result = get_topics("person walking street")
    for topic in result:
        assert "topic_id" in topic
        assert "weight"   in topic
        assert "words"    in topic
        assert isinstance(topic["words"], list)


def test_get_topics_empty_text():
    from src.audio.nlp_analysis import train_lda, get_topics
    texts = ["person walking street", "surfing beach waves",
             "dancing music indoor", "cooking food kitchen",
             "playing soccer field"]
    train_lda(texts, n_topics=3)
    result = get_topics("")
    assert result == []


def test_get_topics_before_training():
    import src.audio.nlp_analysis as m
    m._lda_model      = None
    m._lda_vocabulary = None
    result = m.get_topics("some text here")
    assert result == []


def test_dominant_topic_returns_highest_weight():
    from src.audio.nlp_analysis import train_lda, get_topics, dominant_topic
    texts = [
        "person walking street car road urban",
        "surfing beach waves board water ocean",
        "dancing music indoor floor stage performance",
        "cooking kitchen food eating restaurant meal",
        "playing soccer field sport ball team",
    ]
    train_lda(texts, n_topics=3)
    all_topics = get_topics("person walking street")
    top        = dominant_topic("person walking street")
    assert top["weight"] == all_topics[0]["weight"]


# analyze_transcript 

def test_analyze_transcript_returns_expected_keys():
    from src.audio.nlp_analysis import analyze_transcript
    result = analyze_transcript("A person walking down the street")
    for key in ["keywords", "tfidf_vector", "sentiment",
                "word_count", "sentence_count", "avg_sentence_length"]:
        assert key in result


def test_analyze_transcript_empty():
    from src.audio.nlp_analysis import analyze_transcript
    result = analyze_transcript("")
    assert result["word_count"]    == 0
    assert result["keywords"]      == []
    assert result["sentence_count"] == 0


def test_analyze_transcript_word_count():
    from src.audio.nlp_analysis import analyze_transcript
    result = analyze_transcript("A person is walking down the street today")
    assert result["word_count"] == 8


def test_analyze_transcript_sentiment_present():
    from src.audio.nlp_analysis import analyze_transcript
    result = analyze_transcript("This is great and amazing")
    assert result["sentiment"]["label"] in {"positive", "negative", "neutral"}


def test_analyze_transcript_without_bert():
    from src.audio.nlp_analysis import analyze_transcript
    result = analyze_transcript("A person walking", use_bert=False)
    assert result["bert_embedding"] is None


def test_analyze_transcript_with_bert():
    from src.audio.nlp_analysis import analyze_transcript
    result = analyze_transcript("A person walking down the street", use_bert=True)
    assert result["bert_embedding"] is not None
    assert isinstance(result["bert_embedding"], np.ndarray)
    assert result["bert_embedding"].shape[0] == 384


def test_analyze_transcript_avg_sentence_length():
    from src.audio.nlp_analysis import analyze_transcript
    result = analyze_transcript("She walked. He ran fast.")
    assert result["avg_sentence_length"] > 0