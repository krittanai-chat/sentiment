# simple_sentiment_app.py

import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load the pre-trained model
with open('sentiment_pipeline_model.pkl', 'rb') as file:
    sentiment_pipeline = pickle.load(file)

# Streamlit app interface
st.title("Simple Sentiment Analysis App")
st.write("Enter a sentence to predict its sentiment as Positive or Negative.")

# Text input box for user
user_input = st.text_input("Enter your sentence:")

# Display sentiment prediction
if user_input:
    prediction = sentiment_pipeline.predict([user_input])[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    st.write(f"Sentiment: {sentiment}")

