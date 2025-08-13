# genaifinance
Genaifinance - You can watch the demo here: https://youtu.be/uPWk9Ux0fRc

Overview
Genaifinance is a finance-focused AI application that combines LangChain, OpenAI GPT models, FAISS, and Streamlit to make financial data more accessible and interactive.

Features

Financial News Processing – Load and process financial news articles from given URLs.

Vector Database Search – Store embedded documents in FAISS for fast retrieval.

AI-Powered Q&A – Ask natural language questions about your financial dataset.

Summarization – Get concise AI-generated summaries of large documents.

Interactive UI – Run the app in Streamlit for a clean, interactive experience.

Demo
You can watch the demo here: https://youtu.be/uPWk9Ux0fRc
Watch Video Demo


Installation:

Clone the Repository
git clone https://github.com/fariskajani123/genaifinance.git
cd genaifinance

Create a .env File in the root of the project:
OPENAI_API_KEY=your_openai_api_key_here
Get your API key from: https://platform.openai.com/account/api-keys

Install Dependencies
pip install -r requirements.txt

Run the App
python app.py
Or if using Streamlit:
streamlit run app.py

Usage
Once the app is running:

Enter financial news URLs or upload documents.

Ask AI questions about the content.

Receive answers and summaries in real time.

Project Structure
genaifinance/
├── app.py - Main application entry point
├── GenAi.ipynb - Jupyter notebook for development/testing
├── requirements.txt - Python dependencies
├── .env - API key (ignored in Git)
├── movies.csv - Example dataset
├── sample_text.csv - Sample financial text data
├── nvda_news_1.txt - Example text file
└── README.md - Project documentation

Security Notice

Never commit .env or hardcode API keys.

All keys in this repo are placeholders. Replace with your own before running.

If you leak a key, rotate it immediately.

License
This project is licensed under the MIT License.
