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

# Page config
st.set_page_config(page_title="Consumer Sentiment Dashboard", layout="wide")

st.title("📊 Consumer Sentiment Analysis Dashboard")
st.write("Analyze how people feel about a brand, product, or topic using natural language sentiment analysis.")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file with Tweets or Reviews", type=["csv"])

# Load data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded. Using sample sentiment dataset instead.")
    url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
    df = pd.read_csv(url)
    df = df.rename(columns={'label': 'Sentiment', 'tweet': 'Tweet'})
    df = df[['Tweet', 'Sentiment']]

# Clean up columns
df.columns = df.columns.str.strip().str.capitalize()

# Sentiment Analysis Section
st.subheader("Sentiment Distribution")

# Check if 'Sentiment' column exists, otherwise generate it
if 'Sentiment' not in df.columns:
    # Try to find a text-like column
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
        st.info(f"Detected text column: '{text_col}' — running sentiment analysis...")

        sia = SentimentIntensityAnalyzer()
        df['Sentiment'] = df[text_col].apply(lambda x: (
            'Positive' if sia.polarity_scores(str(x))['compound'] > 0
            else ('Negative' if sia.polarity_scores(str(x))['compound'] < 0
            else 'Neutral')
        ))
else:
    # if sentiment already exists, detect text column for display
    text_col = None
    for col in df.columns:
        if col.lower() in ['text', 'tweet', 'review', 'comment']:
            text_col = col
            break

# Compute sentiment counts
sentiment_counts = df['Sentiment'].value_counts(normalize=True).reset_index()
sentiment_counts.columns = ['Sentiment', 'Percentage']
sentiment_counts['Percentage'] *= 100

# Show chart
fig = px.bar(sentiment_counts,
             x='Sentiment',
             y='Percentage',
             color='Sentiment',
             text=sentiment_counts['Percentage'].apply(lambda x: f"{x:.1f}%"),
             color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'},
             title="Sentiment Breakdown (%)")

fig.update_layout(xaxis_title="", yaxis_title="Percentage", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# Word Cloud
st.subheader("Most Frequent Words in Positive Sentences")

# Automatically detect text column for visualization
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

# Optional: Expandable section for raw data
with st.expander("View Raw Data"):
    st.dataframe(df.head(30))
