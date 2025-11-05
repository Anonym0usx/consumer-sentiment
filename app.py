# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Consumer Sentiment Dashboard", layout="wide")

st.title("📊 Consumer Sentiment Analysis Dashboard")
st.write("Analyze how people feel about a brand, product, or topic.")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file with Tweets or Reviews", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded. Using sample sentiment dataset instead.")
    url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
    df = pd.read_csv(url)
    df = df.rename(columns={'label': 'Sentiment', 'tweet': 'Tweet'})
    df = df[['Tweet', 'Sentiment']]

# Clean column names
df.columns = df.columns.str.strip().str.capitalize()

# Count plot
st.subheader("Sentiment Distribution")
# Check if Sentiment column exists
# Automatically detect and process sentiment
if 'Sentiment' not in df.columns:
    # Try to find columns that contain natural language text
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

        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()

        df['Sentiment'] = df[text_col].apply(lambda x: (
            'Positive' if sia.polarity_scores(str(x))['compound'] > 0
            else ('Negative' if sia.polarity_scores(str(x))['compound'] < 0
            else 'Neutral')
        ))

sentiment_counts = df['Sentiment'].value_counts().reset_index()


# Now safely analyze sentiments
sentiment_counts = df['Sentiment'].value_counts().reset_index()
st.write("Detected columns:", list(df.columns))

# Word cloud for Positive Sentiments
st.subheader("Most Frequent Words in Positive Sentences")
text = ' '.join(df[df['Sentiment'] == 'Positive']['Tweet'].astype(str))
if text:
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
else:
    st.warning("No positive tweets found in the dataset.")

# Optionally show data
with st.expander("View Raw Data"):
    st.dataframe(df.head(20)) 
