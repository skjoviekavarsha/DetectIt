import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load('hate_speech_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in words if word not in stop_words])

# Streamlit interface
st.title("🛡️ Hate Speech Detection App")
st.write("Enter text below to check if it's hate speech.")

user_input = st.text_area("📝 Your Message:")

if st.button("Detect"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        if prediction == 1:
            st.error("⚠️ Hate Speech Detected!")
        else:
            st.success("✅ No Hate Speech Detected.")

st.markdown("---")
st.caption("Made by [Your Name] - Class 12 AI Project")
