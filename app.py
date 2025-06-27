import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load model and vectorizer saved from above training script
model = joblib.load('hate_speech_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

import streamlit as st

st.title("Hate Speech Detection")

user_input = st.text_area("Enter text to classify")

if st.button("Predict"):
    clean_input = clean_text(user_input)
    vect_input = vectorizer.transform([clean_input])
    prediction = model.predict(vect_input)[0]
    
    if prediction == 1:
        st.error("⚠️ Hate Speech Detected!")
    else:
        st.success("✅ No Hate Speech Detected!")


