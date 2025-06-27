import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords (if not already present)
nltk.download('stopwords')

# Load model and vectorizer (make sure these files are in the same folder)
model = joblib.load('hate_speech_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Setup stopwords for preprocessing
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

st.title("🛡️ Hate Speech Detection App")

user_input = st.text_area("Enter text to classify:", height=150)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("P



