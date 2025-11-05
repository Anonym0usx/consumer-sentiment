# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure required data is available
nltk.download('vader_lexicon')

# Detect theme mode (dark/light)
theme = st.get_option("theme.base")
is_dark = theme == "dark"

# Page config
st.set_page_config(page_title="Consumer Sentiment Dashboard", layout="wide")

# Dynamic colors for theme
bg_color = "#0E1117" if is_dark else "#FFFFFF"
text_color = "#FAFAFA" if is_dark else "#333333"
divider_color = "#444" if is_dark else "#CCC"


# Header
st.markdown(
    f"""
    <div style="text-align:center; background-color:{bg_color}; color:{text_color}; padding:10px;">
        <h1>📊 Consumer Sentiment Analysis Dashboard</h1>
        <h3 style="color:gray;">Analyze how people feel about a brand, product, or topic</h3>
        <p style="font-size:15px;">Created by <b>Soham Das</b></p>
        <hr style="border: 1px solid {divider_color};">
    </div>
    """,
    unsafe_allow_html=True
)

# File upload
uploaded_file = st.file_uploader("Upload a CSV file with Tweets or Reviews", type=["csv"])

# Load data
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip', sep=None, engine='python')
else:
    st.info("No file uploaded. Using sample sentiment dataset instead.")
    url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
    df = pd.read_csv(url)
    df = df.rename(columns={'label': 'Sentiment', 'tweet': 'Tweet'})
    df = df[['Tweet', 'Sentiment']]

# Clean up columns
df.columns = df.columns.str.strip().str.capitalize()

# Sentiment Analysis
st.subheader("📈 Sentiment Distribution")

# Check if Sentiment column exists
if 'Sentiment' not in df.columns:
    # Find likely text column
    text_like_cols = []
    for col in df.columns:
        sample_values = df[col].dropna().astype(str).head(10)
        if any(len(v.split()) > 3 for v in sample_values):
            text_like_cols.append(col)

    if not text_like_cols:
        st.error("No text-like column found. Please upload a CSV containing sentences, reviews, or comments.")
        st.stop()
    else:
        text_col = text_like_cols[0]
        # Show progress spinner while analyzing text
        with st.spinner("Analyzing sentiment... please wait"):
            sia = SentimentIntensityAnalyzer()
            df['Sentiment'] = df[text_col].apply(lambda x: (
                'Positive' if sia.polarity_scores(str(x))['compound'] > 0
                else ('Negative' if sia.polarity_scores(str(x))['compound'] < 0
                else 'Neutral')
            ))

                st.success("✅ Sentiment analysis complete!")
else:
    text_col = None
    for col in df.columns:
        if col.lower() in ['text', 'tweet', 'review', 'comment']:
            text_col = col
            break


# Compute sentiment counts
sentiment_counts = df['Sentiment'].value_counts(normalize=True).reset_index()
sentiment_counts.columns = ['Sentiment', 'Percentage']
sentiment_counts['Percentage'] *= 100

# Chart
fig = px.bar(
    sentiment_counts,
    x='Sentiment',
    y='Percentage',
    color='Sentiment',
    text=sentiment_counts['Percentage'].apply(lambda x: f"{x:.1f}%"),
    color_discrete_map={'Positive': '#36AE7C', 'Neutral': '#9BA4B5', 'Negative': '#EB5353'},
    title="Sentiment Breakdown (%)"
)
fig.update_layout(xaxis_title="", yaxis_title="Percentage", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# Quick Stats Summary
st.subheader("📊 Summary Statistics")

# Calculate metrics
total_reviews = len(df)
most_common_sentiment = df['Sentiment'].mode()[0]
avg_text_len = int(df[text_col].astype(str).apply(len).mean()) if text_col else 0

# Create layout
col1, col2, col3 = st.columns(3)
col1.metric("🧾 Total Reviews Analyzed", f"{total_reviews}")
col2.metric("🏆 Most Common Sentiment", f"{most_common_sentiment}")
col3.metric("✏️ Avg Text Length", f"{avg_text_len} chars")


# Summary Metrics
col1, col2, col3 = st.columns(3)
positive = sentiment_counts.loc[sentiment_counts['Sentiment'] == 'Positive', 'Percentage'].sum()
neutral = sentiment_counts.loc[sentiment_counts['Sentiment'] == 'Neutral', 'Percentage'].sum()
negative = sentiment_counts.loc[sentiment_counts['Sentiment'] == 'Negative', 'Percentage'].sum()

col1.metric("😊 Positive", f"{positive:.1f}%")
col2.metric("😐 Neutral", f"{neutral:.1f}%")
col3.metric("☹️ Negative", f"{negative:.1f}%")

# Word Cloud
st.subheader("☁️ Most Frequent Words in Positive Sentences")

if not text_col:
    for col in df.columns:
        if col.lower() in ['text', 'tweet', 'review', 'comment']:
            text_col = col
            break

if text_col:
    text = ' '.join(df[df['Sentiment'] == 'Positive'][text_col].astype(str))
    if text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.warning("No positive text found in the dataset.")
else:
    st.warning("No valid text column found for generating a word cloud.")

# Raw Data Viewer
with st.expander("📄 View Raw Data"):
    st.dataframe(df.head(30))

# Download analyzed data
st.subheader("📥 Download Analyzed Data")

csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="analyzed_sentiment_results.csv",
    mime="text/csv",
    help="Click to download your sentiment analysis results as a CSV file."
)

# Footer + Button Styling
st.markdown("""
    <style>
        div[data-testid="stDownloadButton"] button {
            background-color: #36AE7C;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1.2em;
            font-weight: 500;
            transition: all 0.2s ease-in-out;
        }
        div[data-testid="stDownloadButton"] button:hover {
            background-color: #2E8B6A;
            transform: scale(1.02);
        }
    </style>

    <hr style="border: 0.5px solid #777;">
    <p style="text-align:center; color:gray; font-size:13px;">
    © 2025 Consumer Sentiment Dashboard | Built by <b>Soham Das</b>
    </p>
""", unsafe_allow_html=True)

