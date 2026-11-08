📊 Consumer Sentiment Analysis Dashboard

An interactive Streamlit web app that analyzes how people feel about a brand, product, or topic using natural language sentiment analysis.
Built with Python, NLTK, Plotly, and Streamlit, this dashboard performs real-time sentiment scoring, visualization, and reporting from free-text data such as tweets or product reviews.

🌟 Features

✅ Upload or Use Sample Data
Upload any CSV containing reviews, tweets, or comments — or test the dashboard with a built-in dataset.

✅ Automatic Sentiment Detection
Uses NLTK’s VADER sentiment analyzer to classify text into Positive, Negative, or Neutral categories.

✅ Interactive Visualizations
Dynamic bar charts built with Plotly show the percentage breakdown of sentiment.

✅ Word Cloud Generation
Highlights the most frequent words from positive sentiments for quick keyword insights.

✅ Summary Metrics
Displays total records analyzed, most common sentiment, and average text length.

✅ Download Results
One-click export of analyzed sentiment data as a CSV file.

✅ Dark/Light Mode Adaptive UI
Automatically adjusts colors and styling based on Streamlit’s active theme.

✅ Cached Performance
Both dataset and sentiment analyzer are cached for faster re-runs and smoother interaction.

🧠 Tech Stack

Language: Python 3.x

Frontend Framework: Streamlit

Libraries Used:

pandas, plotly, matplotlib, wordcloud

nltk (VADER Sentiment Analyzer)

Hosting: Streamlit Cloud

Version Control: Git & GitHub

⚙️ How to Run Locally

1. Clone the repository

git clone https://github.com/<your-username>/consumer-sentiment-dashboard.git
cd consumer-sentiment-dashboard


2. Create a virtual environment (recommended)

python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows


3. Install dependencies

pip install -r requirements.txt


If you don’t have a requirements.txt, run this to generate one:

pip freeze > requirements.txt


4. Launch the dashboard

streamlit run app.py


Then open http://localhost:8501
 in your browser.

💡 How to Use the Dashboard

Upload CSV: Choose a CSV containing text data (column names like Text, Tweet, or Review).

View Sentiment Distribution: See the automatic classification chart.

Check Summary Cards: Review total records, common sentiment, and average text length.

Generate Word Cloud: Visualize the most used positive words.

Export Results: Click Download CSV to save labeled results.

🌐 Live Demo

👉 Open the App
https://consumer-sentiment-mczcdix4rbfxzmb9zcun3j.streamlit.app/
(Hosted on Streamlit Cloud - works on desktop and mobile browsers.)

👨‍💻 Author

Soham Das

Passionate about AI, automation, and data-driven decision-making.

📩 LinkedIn: https://www.linkedin.com/in/soham-das-b3a7bb258
 • GitHub: https://github.com/Anonym0usx

🧾 License

This project is open-source and available under the MIT License.

