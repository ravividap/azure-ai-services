"""
Tests for services/openai/chat_completion.py

All Azure SDK calls are mocked so no real credentials are required.
"""

import os
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


def _make_mock_response(content: str):
    """Build a mock AzureOpenAI chat completion response."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def test_chat_returns_assistant_reply():
    from services.openai.chat_completion import chat

    mock_reply = "You should learn Azure OpenAI, Vision, and Language services."
    mock_response = _make_mock_response(mock_reply)

    with patch("services.openai.chat_completion.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        messages = [{"role": "user", "content": "Hello"}]
        result = chat(messages, deployment="gpt-4o")

    assert result == mock_reply
    mock_client.chat.completions.create.assert_called_once()


def test_chat_uses_env_deployment_by_default():
    from services.openai.chat_completion import chat

    mock_response = _make_mock_response("reply")

    with patch("services.openai.chat_completion.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        chat([{"role": "user", "content": "Hi"}])
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"


def test_chat_accepts_custom_deployment():
    from services.openai.chat_completion import chat

    mock_response = _make_mock_response("reply")

    with patch("services.openai.chat_completion.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        chat([{"role": "user", "content": "Hi"}], deployment="gpt-35-turbo")
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-35-turbo"
