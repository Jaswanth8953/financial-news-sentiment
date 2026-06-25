# Financial News Sentiment Dashboard

A real-time dashboard that pulls financial news from major RSS feeds and scores each headline using FinBERT, a BERT-based model fine-tuned on financial text. Built with Streamlit and Plotly for interactive visualization.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![HuggingFace](https://img.shields.io/badge/Model-ProsusAI%2Ffinbert-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green)

## What It Does

The app connects to RSS feeds from MarketWatch, CNBC, and Reuters, fetches the latest headlines, and runs each one through the FinBERT transformer model to produce a sentiment label (positive, negative, or neutral) with a confidence score. Results are displayed as interactive charts alongside a scrollable article list.

## Key Features

- Live RSS ingestion from three major financial news sources
- FinBERT inference cached with `@st.cache_resource` to avoid reloading on every interaction
- Confidence scores and probability breakdown per article (positive/negative/neutral)
- Keyword extraction from headlines using frequency analysis
- Market sentiment index calculation showing overall market mood
- Sentiment distribution visualization with Plotly (pie chart and bar chart)
- Article-level table showing headline, source, sentiment, and confidence
- Custom CSS for a clean, professional dashboard appearance

## Technologies Used

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| ML Model | ProsusAI/FinBERT (Hugging Face Transformers) |
| Deep Learning | PyTorch |
| Dashboard | Streamlit |
| Visualization | Plotly Express, Plotly Graph Objects |
| Data | Feedparser, Pandas, NumPy |

## Getting Started

```bash
git clone https://github.com/Jaswanth8953/financial-news-sentiment.git
cd financial-news-sentiment
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`. Click "Analyze Now" in the sidebar to fetch and score the latest news.

## Project Structure

```
financial-news-sentiment/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md
```

## How It Works

1. The sidebar lists RSS feed URLs targeting financial news sources
2. On button click, the app fetches article titles from each feed using Feedparser
3. All titles are batched and passed to FinBERT for inference
4. The model returns logits converted to softmax probabilities for each label
5. Keyword extraction identifies the most discussed topics across headlines
6. A market sentiment index is calculated from the positive/negative ratio
7. Results feed into Pandas DataFrames for display and download

## Future Improvements

- Add support for user-defined RSS feed URLs via the UI
- Store historical sentiment scores in SQLite for trend analysis over time
- Add a time-series chart showing sentiment shifts across days
- Deploy to Streamlit Cloud with scheduled refresh

## License

MIT
