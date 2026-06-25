# Financial News Sentiment Dashboard

Pulls financial news from 7 RSS feeds and runs each headline through FinBERT to get a positive/negative/neutral label with confidence scores. Built with Streamlit.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![HuggingFace](https://img.shields.io/badge/Model-ProsusAI%2Ffinbert-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green)

## What it does

Fetches articles from MarketWatch, Yahoo Finance, CNN Money, Wall Street Journal, Reuters, and Financial Times RSS feeds. Each headline goes through FinBERT (a BERT model trained on financial text) which returns probabilities for positive, negative, and neutral sentiment. The results feed into four tabs:

- **Dashboard** - pie chart of sentiment distribution, gauge showing an overall market sentiment index (calculated as (positive - negative) / total * 100)
- **Articles** - filterable list of headlines with source, confidence score, and a link to read the full article
- **Keywords** - top 20 words extracted from headlines, shown as a bar chart and a treemap
- **Download** - export the full results as a CSV

## Setup

```bash
git clone https://github.com/Jaswanth8953/financial-news-sentiment.git
cd financial-news-sentiment
pip install -r requirements.txt
streamlit run app.py
```

Runs on CPU by default. First run downloads the FinBERT weights from Hugging Face (~400MB). After that they're cached locally.

## Stack

| What | How |
|---|---|
| NLP model | ProsusAI/FinBERT via Hugging Face Transformers |
| Inference | PyTorch (CPU) |
| Dashboard | Streamlit |
| Charts | Plotly Express + Plotly Graph Objects (gauge) |
| Feed parsing | feedparser |
| Data | pandas, numpy |

## Notes

- The model is loaded once with `@st.cache_resource` so it stays in memory across reruns
- Sentiment probabilities come from `torch.softmax` on the raw logits, not just argmax
- Keywords tab strips stop words and counts remaining words with `collections.Counter`
- The requirements.txt uses the PyTorch CPU wheel to keep the install size manageable

## License

MIT
