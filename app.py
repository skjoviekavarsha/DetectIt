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
        st.warning("Please enter some text to analyze.")
    else:
        clean_input = clean_text(user_input)
        vect_input = vectorizer.transform([clean_input])
        prediction = model.predict(vect_input)[0]
        prob = model.predict_proba(vect_input)[0]

        st.markdown(f"**Processed text:** {clean_input}")
        st.markdown(f"**Prediction:** {'⚠️ Hate Speech Detected!' if prediction == 1 else '✅ No Hate Speech Detected!'}")
        st.markdown(f"**Probabilities:** Not Hate Speech: {prob[0]:.3f}, Hate Speech: {prob[1]:.3f}")



