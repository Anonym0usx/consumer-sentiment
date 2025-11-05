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
sentiment_counts = df['Sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']
fig = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment',
             color_discrete_map={'Positive':'green','Negative':'red','Neutral':'gray'})
st.plotly_chart(fig, use_container_width=True)

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
