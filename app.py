import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

model = joblib.load('hate_speech_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("Hate Speech Detection")

text = st.text_area("Enter text to classify:")

if st.button("Predict"):
    cleaned = clean_text(text)
    vect_text = vectorizer.transform([cleaned])
    prediction = model.predict(vect_text)

    if prediction[0] == 1:
        st.error("⚠️ Hate Speech Detected!")
    else:
        st.success("✅ The text is NOT hate speech.")

