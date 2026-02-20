# Azure AI Services – Python Practice

Hands-on Python examples for AI engineering with **Azure AI Studio / Azure AI Services**.
Each service lives in its own module under `services/` and is fully runnable once you
supply your Azure credentials.

---

## Services covered

| # | Service | Module | What it shows |
|---|---------|--------|---------------|
| 1 | **Azure OpenAI** | `services/openai/chat_completion.py` | Chat completion with `gpt-4o` |
| 2 | **Azure AI Vision** | `services/vision/image_analysis.py` | Image captions, tags & object detection |
| 3 | **Azure AI Language** | `services/language/text_analytics.py` | Sentiment analysis, key phrases & NER |

---

## Quick start

### 1 – Clone & install dependencies

```bash
git clone https://github.com/ravividap/azure-ai-services.git
cd azure-ai-services
pip install -r requirements.txt
```

### 2 – Configure credentials

Copy `.env.example` to `.env` and fill in your Azure resource endpoints and keys:

```bash
cp .env.example .env
# edit .env with your values
```

### 3 – Run an example

```bash
# Azure OpenAI – chat completion
python services/openai/chat_completion.py

# Azure AI Vision – image analysis
python services/vision/image_analysis.py

# Azure AI Language – text analytics
python services/language/text_analytics.py
```

---

## Project structure

```
azure-ai-services/
├── services/
│   ├── openai/
│   │   └── chat_completion.py   # Azure OpenAI chat completion
│   ├── vision/
│   │   └── image_analysis.py    # Azure AI Vision image analysis
│   └── language/
│       └── text_analytics.py    # Azure AI Language text analytics
├── tests/
│   ├── test_chat_completion.py
│   ├── test_image_analysis.py
│   └── test_text_analytics.py
├── .env.example                 # Environment variable template
├── requirements.txt
└── README.md
```

---

## Running tests

Tests use `pytest` with mocked Azure SDK calls – no real credentials needed.

```bash
pytest tests/ -v
```

---

## Prerequisites

* Python 3.10+
* An [Azure subscription](https://azure.microsoft.com/free/)
* Provisioned resources for each service you want to use (see `.env.example`)
