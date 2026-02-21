"""
Azure AI Vision – Image Analysis example.

Demonstrates how to analyze an image (from a public URL or a local file)
using the Azure AI Vision 4.x REST endpoint via the
``azure-cognitiveservices-vision-computervision`` SDK.

Usage:
    python services/vision/image_analysis.py
"""

import os
from dotenv import load_dotenv
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

load_dotenv()

# Default public image used when running the demo standalone.
_DEMO_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/"
    "Cat03.jpg/1200px-Cat03.jpg"
)


def get_client() -> ComputerVisionClient:
    """Create and return a ComputerVisionClient."""
    endpoint = os.environ["AZURE_VISION_ENDPOINT"]
    key = os.environ["AZURE_VISION_KEY"]
    credentials = CognitiveServicesCredentials(key)
    return ComputerVisionClient(endpoint, credentials)


def analyze_image_url(image_url: str) -> dict:
    """
    Analyze a publicly accessible image and return structured results.

    Args:
        image_url: Public URL of the image to analyze.

    Returns:
        A dict containing 'description', 'tags', and 'objects' lists.
    """
    client = get_client()
    features = [
        VisualFeatureTypes.description,
        VisualFeatureTypes.tags,
        VisualFeatureTypes.objects,
    ]
    result = client.analyze_image(image_url, visual_features=features)

    captions = [
        {"text": c.text, "confidence": round(c.confidence, 4)}
        for c in (result.description.captions or [])
    ]
    tags = [
        {"name": t.name, "confidence": round(t.confidence, 4)}
        for t in (result.tags or [])
    ]
    objects = [
        {
            "object": o.object_property,
            "confidence": round(o.confidence, 4),
            "rectangle": {
                "x": o.rectangle.x,
                "y": o.rectangle.y,
                "w": o.rectangle.w,
                "h": o.rectangle.h,
            },
        }
        for o in (result.objects or [])
    ]

    return {"captions": captions, "tags": tags, "objects": objects}


def main() -> None:
    print("=== Azure AI Vision – Image Analysis ===\n")
    print(f"Analyzing image: {_DEMO_IMAGE_URL}\n")

    analysis = analyze_image_url(_DEMO_IMAGE_URL)

    print("Captions:")
    for caption in analysis["captions"]:
        print(f"  • {caption['text']}  (confidence: {caption['confidence']:.2%})")

    print("\nTop Tags:")
    for tag in analysis["tags"][:8]:
        print(f"  • {tag['name']}  (confidence: {tag['confidence']:.2%})")

    print("\nDetected Objects:")
    for obj in analysis["objects"]:
        print(f"  • {obj['object']}  (confidence: {obj['confidence']:.2%})")


if __name__ == "__main__":
    main()
