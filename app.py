import pandas as pd
import joblib
import nltk
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load dataset
url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
df = pd.read_csv(url)

# Combine 'hate' and 'offensive' into one class: 1
df['label'] = df['class'].apply(lambda x: 1 if x in [1, 2] else 0)

# Balance the dataset by downsampling majority class (label 0)
count_class_0 = df[df['label'] == 0].shape[0]
count_class_1 = df[df['label'] == 1].shape[0]

if count_class_0 > count_class_1:
    df_class_0 = df[df['label'] == 0].sample(count_class_1, random_state=42)
    df_class_1 = df[df['label'] == 1]
    df_balanced = pd.concat([df_class_0, df_class_1])
else:
    df_balanced = df  # If already balanced

# --- DEBUG PRINTS START HERE ---
print("Examples of NOT hate speech (label 0):")
print(df_balanced[df_balanced['label'] == 0]['tweet'].sample(5).tolist())

print("\nExamples of Hate speech (label 1):")
print(df_balanced[df_balanced['label'] == 1]['tweet'].sample(5).tolist())
# --- DEBUG PRINTS END HERE ---

# Preprocessing function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

df_balanced['clean_text'] = df_balanced['tweet'].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_balanced['clean_text'])
y = df_balanced['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training with balanced class weights
model = LogisticRegression(max_iter=1000, class_weight='balanced')
print("Label counts after balancing:")
print(df_balanced['label'].value_counts())
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Test prediction on "great"
test_vect = vectorizer.transform(['great'])
pred = model.predict(test_vect)[0]
proba = model.predict_proba(test_vect)[0]
print(f"\nPrediction for 'great': {pred}")
print(f"Probabilities for 'great': Not Hate Speech: {proba[0]:.3f}, Hate Speech: {proba[1]:.3f}")

# Save model and vectorizer
joblib.dump(model, 'hate_speech_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("\n✅ Model and vectorizer saved successfully!")
