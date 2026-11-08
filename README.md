# CXInsight

**Customer Experience Intelligence Platform**

CXInsight is a customer feedback analytics application that helps transform unstructured customer reviews into understandable business insights. Upload a CSV file to explore sentiment, discover recurring feedback themes, review customer comments, and ask an AI assistant questions about aggregated results.

## Features

- CSV upload and processing.
- Automatic feedback-column detection.
- VADER sentiment classification: Positive, Negative, or Neutral.
- Topic discovery using TF-IDF and K-Means.
- Business overview: feedback volume, sentiment percentages, average rating (when available), and NPS (when available).
- Topic-level sentiment comparison.
- Feedback Explorer with search and sentiment/topic filters.
- CSV export of filtered feedback.
- CXInsight AI powered by Cloudflare Workers AI. The application provides aggregated analysis context rather than the complete raw review dataset.

## Technology stack

Python · Gradio · Pandas · NumPy · Scikit-learn · VADER Sentiment · Plotly · Cloudflare Workers AI

## Project structure

```text
CXInsights/
├── gradio_app.py
├── requirements.txt
├── README.md
├── .gitignore
├── static/
│   └── styles.css
├── src/
│   ├── __init__.py
│   ├── analytics.py
│   ├── column_detector.py
│   ├── data_processor.py
│   ├── sentiment_analyzer.py
│   ├── topic_analyzer.py
│   └── cloudflare_llm.py
├── tests/
│   ├── test_analytics.py
│   ├── test_column_detector.py
│   ├── test_data_processor.py
│   ├── test_sentiment.py
│   └── test_topics.py
└── data/
    └── sample_feedback.csv
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/razzaq-ml-ds/CXInsight.git
cd CXInsight
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv-gradio
.\.venv-gradio\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Cloudflare Workers AI

The AI assistant requires Cloudflare Workers AI credentials. Create the local file `.streamlit/secrets.toml`:

```toml
[cloudflare]
account_id = "YOUR_CLOUDFLARE_ACCOUNT_ID"
api_token = "YOUR_CLOUDFLARE_API_TOKEN"
model = "@cf/zai-org/glm-4.7-flash"
```

Alternatively, set environment variables:

```powershell
$env:CLOUDFLARE_ACCOUNT_ID = "YOUR_CLOUDFLARE_ACCOUNT_ID"
$env:CLOUDFLARE_API_TOKEN = "YOUR_CLOUDFLARE_API_TOKEN"
$env:CLOUDFLARE_MODEL = "@cf/zai-org/glm-4.7-flash"
```

Never commit API tokens or `.streamlit/secrets.toml` to Git. The analytics dashboard can be used without the AI assistant; AI features require valid Cloudflare credentials and network access.

### 5. Run the application

```powershell
python gradio_app.py
```

Open `http://127.0.0.1:7860` in your browser. Keep the terminal running while using the application.

## Running the tests

Run commands from the repository root. In PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python tests/test_analytics.py
python tests/test_column_detector.py
python tests/test_data_processor.py
python tests/test_sentiment.py
python tests/test_topics.py
```

These are script-based backend checks using the sample dataset and assertions where currently implemented.

## Input data

Upload a CSV containing customer feedback. CXInsight attempts to detect the feedback-text column automatically. A rating column can also be used for rating and NPS-related metrics when the expected data is present. A sample dataset is included for local testing.

## Customize the interface

The interface stylesheet is maintained separately at `static/styles.css`. Edit it to adjust colors, typography, spacing, and layout, then restart Gradio to load the changes.

## Security and responsible use

- Keep Cloudflare credentials outside source control.
- Upload customer data only when you are authorized to do so.
- AI-generated insights and sentiment labels are estimates. Review them before making consequential business decisions.

## Project scope

CXInsight is a portfolio project demonstrating a Python customer-feedback analytics workflow and interactive Gradio interface.

---

Built with Python, data analysis, and machine learning.
