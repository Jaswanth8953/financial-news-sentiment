import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import feedparser
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datetime import datetime
import time
from collections import Counter
import re
import numpy as np

# Load model once and cache it
@st.cache_resource
def get_model():
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    return model, tokenizer

# Get news from RSS feed
def get_news_from_feed(url):
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries:
            description = ""
            if hasattr(entry, 'summary'):
                description = entry.summary[:200]
            elif hasattr(entry, 'description'):
                description = entry.description[:200]
            
            pub_date = ""
            if hasattr(entry, 'published'):
                pub_date = entry.published
            elif hasattr(entry, 'updated'):
                pub_date = entry.updated
            
            articles.append({
                'title': entry.title,
                'link': entry.link,
                'source': url,
                'description': description,
                'published': pub_date
            })
        return articles
    except:
        return []

# Analyze sentiment with detailed probabilities
def check_sentiment(articles, model, tokenizer):
    if not articles:
        return articles

    titles = [a['title'] for a in articles]
    inputs = tokenizer(titles, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        output = model(**inputs)

    # Get sentiment scores for confidence
    logits = output.logits
    probs = torch.softmax(logits, dim=1)
    sentiments = torch.argmax(logits, dim=1)
    labels = ['positive', 'negative', 'neutral']

    for i, article in enumerate(articles):
        article['sentiment'] = labels[sentiments[i]]
        article['confidence'] = round(float(probs[i].max().item()) * 100, 1)
        article['positive_score'] = round(float(probs[i][0].item()) * 100, 1)
        article['negative_score'] = round(float(probs[i][1].item()) * 100, 1)
        article['neutral_score'] = round(float(probs[i][2].item()) * 100, 1)

    return articles

# Extract keywords from titles
def extract_keywords(titles, top_n=15):
    words = []
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are'}
    
    for title in titles:
        # Remove special characters and split
        cleaned = re.sub(r'[^\w\s]', '', title.lower())
        title_words = [w for w in cleaned.split() if w not in stop_words and len(w) > 3]
        words.extend(title_words)
    
    return Counter(words).most_common(top_n)

# Calculate market sentiment index
def calculate_sentiment_index(df):
    pos = len(df[df['sentiment'] == 'positive'])
    neg = len(df[df['sentiment'] == 'negative'])
    total = len(df)
    
    if total == 0:
        return 0
    
    index = ((pos - neg) / total) * 100
    return round(index, 2)


st.set_page_config(page_title="Financial News Sentiment Analysis", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for professional look
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header-title {
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Financial News Sentiment Analysis")
st.markdown("*Real-time market sentiment analysis powered by FinBERT*")

# Sidebar setup
st.sidebar.header("Settings")

# RSS feeds list
feeds = [
    'http://feeds.marketwatch.com/marketwatch/topstories/',
    'http://feeds.marketwatch.com/marketwatch/marketpulse/',
    'https://feeds.finance.yahoo.com/rss/2.0/headline',
    'http://rss.cnn.com/rss/money_latest.rss',
    'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',  # Wall Street Journal
    'https://feeds.reuters.com/reuters/businessNews',
    'https://www.ft.com/rss/home/us',  # Financial Times
]

st.sidebar.write(f"Analyzing {len(feeds)} RSS feeds")

# Add refresh timestamp
col1, col2 = st.sidebar.columns(2)
with col1:
    last_run = st.sidebar.empty()
with col2:
    refresh_time = st.sidebar.empty()

# Main app
if st.sidebar.button("Analyze Now"):
    st.write("Loading model...")
    model, tokenizer = get_model() #finbert

    all_articles = []

    # Get articles from all feeds
    st.write("Fetching articles...")
    for feed_url in feeds:
        articles = get_news_from_feed(feed_url)
        all_articles.extend(articles)

    # Check sentiment
    st.write(f"Analyzing {len(all_articles)} articles...")
    articles_with_sentiment = check_sentiment(all_articles, model, tokenizer)

    # Make dataframe
    df = pd.DataFrame(articles_with_sentiment)
    
    # Add timestamp
    df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate sentiment index
    sentiment_index = calculate_sentiment_index(df)

    # Show KPI metrics
    st.markdown("---")
    st.subheader("Market Sentiment Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)

    pos = len(df[df['sentiment'] == 'positive'])
    neg = len(df[df['sentiment'] == 'negative'])
    neu = len(df[df['sentiment'] == 'neutral'])
    avg_conf = df['confidence'].mean()

    col1.metric("Total Articles", len(df), delta=None)
    col2.metric("Positive", pos, delta=f"{(pos/len(df)*100):.1f}%")
    col3.metric("Negative", neg, delta=f"{(neg/len(df)*100):.1f}%")
    col4.metric("Neutral", neu, delta=f"{(neu/len(df)*100):.1f}%")
    
    # Sentiment Index with color coding
    if sentiment_index > 0:
        col5.metric("Sentiment Index", f"{sentiment_index:.1f}", delta=f"Bullish", delta_color="inverse")
    else:
        col5.metric("Sentiment Index", f"{sentiment_index:.1f}", delta=f"Bearish", delta_color="inverse")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Articles", "Keywords", "Download"])

    with tab1:
        st.subheader("Executive Dashboard")
        
        col1, col2 = st.columns(2)

        with col1:
            # Pie chart with improved styling
            sentiment_count = df['sentiment'].value_counts()
            fig = px.pie(names=sentiment_count.index, values=sentiment_count.values,
                        title="Sentiment Distribution",
                        color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'},
                        hole=0.3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Gauge chart for sentiment index
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=sentiment_index,
                title={'text': "Sentiment Index"},
                delta={'reference': 0},
                gauge={
                    'axis': {'range': [-100, 100]},
                    'bar': {'color': "#2ecc71" if sentiment_index > 0 else "#e74c3c"},
                    'steps': [
                        {'range': [-100, 0], 'color': "rgba(231, 76, 60, 0.2)"},
                        {'range': [0, 100], 'color': "rgba(46, 204, 113, 0.2)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Source and confidence analysis
        col1, col2 = st.columns(2)
        
        with col1:
            source_counts = df['source'].str.extract(r'https?://([^/]+)')[0].value_counts()
            fig = px.bar(x=source_counts.values, y=source_counts.index, orientation='h',
                        title="Articles by Source",
                        labels={'x': 'Count', 'y': 'Source'},
                        color=source_counts.values,
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            avg_conf = df.groupby('sentiment')['confidence'].mean()
            fig = px.bar(x=avg_conf.index, y=avg_conf.values,
                        title="Average Confidence by Sentiment",
                        color=avg_conf.index,
                        color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'},
                        labels={'y': 'Confidence %', 'x': 'Sentiment'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.write("### Articles")

        # Filter by sentiment
        sentiment_filter = st.multiselect("Filter by sentiment",
                                         options=['positive', 'negative', 'neutral'],
                                         default=['positive', 'negative', 'neutral'])

        filtered = df[df['sentiment'].isin(sentiment_filter)].sort_values('confidence', ascending=False)

        for idx, row in filtered.iterrows():
            source_name = row['source'].replace('http://', '').replace('https://', '').split('/')[0]
            sentiment_color = {'positive': '▲', 'negative': '▼', 'neutral': '–'}[row['sentiment']]
            
            st.write(f"{sentiment_color} **{row['title'][:100]}**")
            
            col1, col2, col3 = st.columns(3)
            col1.caption(f"{source_name}")
            col2.caption(f"Confidence: {row['confidence']}%")
            col3.caption(f"[Read →]({row['link']})")
            
            if row['description']:
                st.caption(f"_{row['description']}_")
            
            st.write("")

    with tab3:
        st.subheader("Trending Topics")
        
        keywords = extract_keywords(df['title'].tolist(), top_n=20)
        if keywords:
            keywords_df = pd.DataFrame(keywords, columns=['Keyword', 'Frequency'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(keywords_df, x='Frequency', y='Keyword', orientation='h',
                           title="Top Keywords in News",
                           color='Frequency',
                           color_continuous_scale='Blues')
                fig.update_yaxes(categoryorder='total ascending')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Word cloud alternative - treemap
                fig = px.treemap(keywords_df, labels='Keyword', parents=[''] * len(keywords_df), values='Frequency',
                               title="Keywords Treemap",
                               color='Frequency',
                               color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        # Download
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, file_name="sentiment.csv", mime="text/csv")

        # Show table
        st.write("### Data")
        st.dataframe(df, use_container_width=True)
