import streamlit as st
import pickle
from scipy.sparse import hstack

# Load saved model and vectorizers
model = pickle.load(open("model.pkl", "rb"))
tfidf_title = pickle.load(open("tfidf_title.pkl", "rb"))
tfidf_text = pickle.load(open("tfidf_text.pkl", "rb"))

# App Title
st.title("📰 Fake News Detection")

st.write("Enter news title and content to check whether it is Real or Fake.")

# User Input
title = st.text_input("News Title")
text = st.text_area("News Content")

# Prediction Button
if st.button("Check News"):
    
    if title and text:
        # Convert text to numbers
        title_vector = tfidf_title.transform([title])
        text_vector = tfidf_text.transform([text])

        # Combine both
        final_input = hstack([title_vector, text_vector])

        # Predict
        prediction = model.predict(final_input)[0]

        # Show result
        if prediction == 1:
            st.success("✅ This is Real News")
        else:
            st.error("❌ This is Fake News")
    
    else:
        st.warning("Please enter both title and content.")