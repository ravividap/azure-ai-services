"""
Tests for services/language/text_analytics.py

All Azure SDK calls are mocked so no real credentials are required.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("AZURE_LANGUAGE_ENDPOINT", "https://test.cognitiveservices.azure.com/")
    monkeypatch.setenv("AZURE_LANGUAGE_KEY", "test-key")


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_sentiment_result(sentiment: str, pos: float, neu: float, neg: float):
    scores = MagicMock()
    scores.positive = pos
    scores.neutral = neu
    scores.negative = neg
    result = MagicMock()
    result.is_error = False
    result.sentiment = sentiment
    result.confidence_scores = scores
    return result


def _make_key_phrases_result(phrases: list[str]):
    result = MagicMock()
    result.is_error = False
    result.key_phrases = phrases
    return result


def _make_entity(text: str, category: str, confidence: float):
    e = MagicMock()
    e.text = text
    e.category = category
    e.confidence_score = confidence
    return e


def _make_entities_result(entities):
    result = MagicMock()
    result.is_error = False
    result.entities = entities
    return result


# ── sentiment ────────────────────────────────────────────────────────────────

def test_analyze_sentiment_positive():
    from services.language.text_analytics import analyze_sentiment

    mock_result = _make_sentiment_result("positive", 0.98, 0.01, 0.01)

    with patch("services.language.text_analytics.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.analyze_sentiment.return_value = [mock_result]
        mock_get_client.return_value = mock_client

        output = analyze_sentiment(["Azure AI is great!"])

    assert output[0]["sentiment"] == "positive"
    assert output[0]["scores"]["positive"] == round(0.98, 4)


def test_analyze_sentiment_negative():
    from services.language.text_analytics import analyze_sentiment

    mock_result = _make_sentiment_result("negative", 0.02, 0.03, 0.95)

    with patch("services.language.text_analytics.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.analyze_sentiment.return_value = [mock_result]
        mock_get_client.return_value = mock_client

        output = analyze_sentiment(["This is terrible."])

    assert output[0]["sentiment"] == "negative"


def test_analyze_sentiment_error_propagated():
    from services.language.text_analytics import analyze_sentiment

    mock_result = MagicMock()
    mock_result.is_error = True
    mock_result.error.message = "Invalid document"

    with patch("services.language.text_analytics.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.analyze_sentiment.return_value = [mock_result]
        mock_get_client.return_value = mock_client

        output = analyze_sentiment([""])

    assert "error" in output[0]
    assert output[0]["error"] == "Invalid document"


# ── key phrases ───────────────────────────────────────────────────────────────

def test_extract_key_phrases_returns_list():
    from services.language.text_analytics import extract_key_phrases

    mock_result = _make_key_phrases_result(["Azure AI", "Python", "cognitive services"])

    with patch("services.language.text_analytics.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.extract_key_phrases.return_value = [mock_result]
        mock_get_client.return_value = mock_client

        output = extract_key_phrases(["Azure AI is great for Python developers."])

    assert "Azure AI" in output[0]["key_phrases"]
    assert "Python" in output[0]["key_phrases"]


# ── named entity recognition ──────────────────────────────────────────────────

def test_recognize_entities_returns_structured_list():
    from services.language.text_analytics import recognize_entities

    entities = [
        _make_entity("Microsoft", "Organization", 0.99),
        _make_entity("Azure", "Product", 0.97),
    ]
    mock_result = _make_entities_result(entities)

    with patch("services.language.text_analytics.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.recognize_entities.return_value = [mock_result]
        mock_get_client.return_value = mock_client

        output = recognize_entities(["Microsoft Azure is a cloud platform."])

    entity_texts = [e["text"] for e in output[0]["entities"]]
    assert "Microsoft" in entity_texts
    assert "Azure" in entity_texts
    assert output[0]["entities"][0]["category"] == "Organization"
