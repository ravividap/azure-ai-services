"""
Azure AI Language – Text Analytics example.

Demonstrates sentiment analysis, key phrase extraction, and named entity
recognition (NER) using the ``azure-ai-textanalytics`` SDK.

Usage:
    python services/language/text_analytics.py
"""

import os
from dotenv import load_dotenv
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()

_DEMO_DOCUMENTS = [
    "Azure AI services are incredibly powerful and easy to integrate into Python apps.",
    "I had a terrible experience with the slow API response times last week.",
    "Microsoft's cognitive services include vision, language, and speech capabilities.",
]


def get_client() -> TextAnalyticsClient:
    """Create and return a TextAnalyticsClient."""
    endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
    key = os.environ["AZURE_LANGUAGE_KEY"]
    return TextAnalyticsClient(endpoint, AzureKeyCredential(key))


def analyze_sentiment(documents: list[str]) -> list[dict]:
    """
    Perform sentiment analysis on a list of documents.

    Args:
        documents: Plain-text strings to analyze.

    Returns:
        List of dicts with 'text', 'sentiment', and 'scores' keys.
    """
    client = get_client()
    results = client.analyze_sentiment(documents)
    output = []
    for doc, result in zip(documents, results):
        if result.is_error:
            output.append({"text": doc, "error": result.error.message})
        else:
            output.append(
                {
                    "text": doc,
                    "sentiment": result.sentiment,
                    "scores": {
                        "positive": round(result.confidence_scores.positive, 4),
                        "neutral": round(result.confidence_scores.neutral, 4),
                        "negative": round(result.confidence_scores.negative, 4),
                    },
                }
            )
    return output


def extract_key_phrases(documents: list[str]) -> list[dict]:
    """
    Extract key phrases from a list of documents.

    Args:
        documents: Plain-text strings to process.

    Returns:
        List of dicts with 'text' and 'key_phrases' keys.
    """
    client = get_client()
    results = client.extract_key_phrases(documents)
    output = []
    for doc, result in zip(documents, results):
        if result.is_error:
            output.append({"text": doc, "error": result.error.message})
        else:
            output.append({"text": doc, "key_phrases": result.key_phrases})
    return output


def recognize_entities(documents: list[str]) -> list[dict]:
    """
    Recognize named entities in a list of documents.

    Args:
        documents: Plain-text strings to process.

    Returns:
        List of dicts with 'text' and 'entities' keys.
    """
    client = get_client()
    results = client.recognize_entities(documents)
    output = []
    for doc, result in zip(documents, results):
        if result.is_error:
            output.append({"text": doc, "error": result.error.message})
        else:
            entities = [
                {
                    "text": e.text,
                    "category": e.category,
                    "confidence": round(e.confidence_score, 4),
                }
                for e in result.entities
            ]
            output.append({"text": doc, "entities": entities})
    return output


def main() -> None:
    print("=== Azure AI Language – Text Analytics ===\n")

    print("-- Sentiment Analysis --")
    for item in analyze_sentiment(_DEMO_DOCUMENTS):
        if "error" in item:
            print(f"  ERROR: {item['error']}")
        else:
            scores = item["scores"]
            print(
                f"  [{item['sentiment'].upper():8}] "
                f"+{scores['positive']:.2f} / ~{scores['neutral']:.2f} / -{scores['negative']:.2f}"
                f"\n           \"{item['text'][:70]}\""
            )

    print("\n-- Key Phrase Extraction --")
    for item in extract_key_phrases(_DEMO_DOCUMENTS):
        if "error" in item:
            print(f"  ERROR: {item['error']}")
        else:
            print(f"  Phrases: {', '.join(item['key_phrases'])}")

    print("\n-- Named Entity Recognition --")
    for item in recognize_entities(_DEMO_DOCUMENTS):
        if "error" in item:
            print(f"  ERROR: {item['error']}")
        else:
            for e in item["entities"]:
                print(f"  • {e['text']} [{e['category']}] (confidence: {e['confidence']:.2%})")
            print()


if __name__ == "__main__":
    main()
