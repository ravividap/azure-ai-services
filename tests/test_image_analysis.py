"""
Tests for services/vision/image_analysis.py

All Azure SDK calls are mocked so no real credentials are required.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("AZURE_VISION_ENDPOINT", "https://test.cognitiveservices.azure.com/")
    monkeypatch.setenv("AZURE_VISION_KEY", "test-key")


def _make_caption(text: str, confidence: float):
    c = MagicMock()
    c.text = text
    c.confidence = confidence
    return c


def _make_tag(name: str, confidence: float):
    t = MagicMock()
    t.name = name
    t.confidence = confidence
    return t


def _make_object(name: str, confidence: float):
    rect = MagicMock()
    rect.x, rect.y, rect.w, rect.h = 10, 20, 100, 80
    o = MagicMock()
    o.object_property = name
    o.confidence = confidence
    o.rectangle = rect
    return o


def _build_mock_result():
    description = MagicMock()
    description.captions = [_make_caption("a cat sitting on a sofa", 0.95)]

    result = MagicMock()
    result.description = description
    result.tags = [_make_tag("cat", 0.99), _make_tag("animal", 0.97)]
    result.objects = [_make_object("cat", 0.95)]
    return result


def test_analyze_image_url_returns_structured_dict():
    from services.vision.image_analysis import analyze_image_url

    mock_result = _build_mock_result()

    with patch("services.vision.image_analysis.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.analyze_image.return_value = mock_result
        mock_get_client.return_value = mock_client

        output = analyze_image_url("https://example.com/cat.jpg")

    assert "captions" in output
    assert "tags" in output
    assert "objects" in output
    assert output["captions"][0]["text"] == "a cat sitting on a sofa"
    assert output["captions"][0]["confidence"] == round(0.95, 4)


def test_analyze_image_url_maps_tags():
    from services.vision.image_analysis import analyze_image_url

    mock_result = _build_mock_result()

    with patch("services.vision.image_analysis.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.analyze_image.return_value = mock_result
        mock_get_client.return_value = mock_client

        output = analyze_image_url("https://example.com/cat.jpg")

    tag_names = [t["name"] for t in output["tags"]]
    assert "cat" in tag_names
    assert "animal" in tag_names


def test_analyze_image_url_maps_objects():
    from services.vision.image_analysis import analyze_image_url

    mock_result = _build_mock_result()

    with patch("services.vision.image_analysis.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.analyze_image.return_value = mock_result
        mock_get_client.return_value = mock_client

        output = analyze_image_url("https://example.com/cat.jpg")

    assert output["objects"][0]["object"] == "cat"
    assert "rectangle" in output["objects"][0]
