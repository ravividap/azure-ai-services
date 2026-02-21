"""
Azure OpenAI Service – Chat Completion example.

Demonstrates how to send a prompt to an Azure OpenAI deployment and
stream the response back to the console.

Usage:
    python services/openai/chat_completion.py
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()


def get_client() -> AzureOpenAI:
    """Create and return an AzureOpenAI client."""
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-02-01",
    )


def chat(messages: list[dict], deployment: str | None = None) -> str:
    """
    Send *messages* to an Azure OpenAI chat model and return the reply.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        deployment: Azure deployment name.  Defaults to the
                    ``AZURE_OPENAI_DEPLOYMENT`` environment variable.

    Returns:
        The assistant reply as a plain string.
    """
    deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    client = get_client()

    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].message.content


def main() -> None:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI engineering tutor who explains Azure AI "
                "services clearly and concisely."
            ),
        },
        {
            "role": "user",
            "content": "What are three Azure AI services I should learn first and why?",
        },
    ]

    print("=== Azure OpenAI – Chat Completion ===\n")
    reply = chat(messages)
    print(reply)


if __name__ == "__main__":
    main()
