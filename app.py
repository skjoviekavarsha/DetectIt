import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load('hate_speech_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


# Streamlit App UI
st.title("🛡️ Hate Speech Detection App")
st.write("Enter text to classify:")

threshold = st.slider("Set hate speech threshold", 0.0, 1.0, 0.7)

user_input = st.text_area("Enter text to classify:")

if st.button("Detect"):
    cleaned_input = clean_text(user_input)
    vector = vectorizer.transform([cleaned_input])
    proba = model.predict_proba(vector)[0]
    prediction = 1 if proba[1] > threshold else 0  # Use slider value here

    st.write(f"**Processed text:** {cleaned_input}")
    
    if prediction == 1:
        st.error("⚠️ Hate Speech Detected!")
    else:
        st.success("✅ Not Hate Speech")

    st.write(f"**Probabilities:** Not Hate Speech: `{proba[0]:.3f}`, Hate Speech: `{proba[1]:.3f}`")
