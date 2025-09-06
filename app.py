import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import os

# ===========================
# Text cleaning function
# ===========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# ===========================
# Load or initialize model/vectorizer
# ===========================
model_path = "spam_model.pkl"
vectorizer_path = "vectorizer.pkl"

model = joblib.load(model_path) if os.path.exists(model_path) else None
vectorizer = joblib.load(vectorizer_path) if os.path.exists(vectorizer_path) else None

# ===========================
# Streamlit UI
# ===========================
st.set_page_config(page_title="Spam Classifier", layout="centered")
st.title("📧 Email Spam Classifier")

# --- Upload dataset and retrain
st.header("📂 Upload Dataset and Retrain")

uploaded_file = st.file_uploader("Upload your CSV file (must include 'title', 'text', and 'type')", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if all(col in df.columns for col in ['title', 'text', 'type']):
        st.success("✅ File loaded successfully.")
        df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        df['label'] = df['type'].map({'spam': 1, 'not spam': 0})
        df['content'] = df['content'].apply(clean_text)

        vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
        X = vectorizer.fit_transform(df['content'])
        y = df['label']

        sm = SMOTE(random_state=42)
        X_resampled, y_resampled = sm.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        model = MultinomialNB()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(f"Model trained! ✅ Accuracy: {acc:.2%}")
        st.text("Classification Report:")
        st.code(classification_report(y_test, y_pred), language='text')

        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        st.info("Model and vectorizer saved.")
    else:
        st.error("❌ Your CSV must have columns: 'title', 'text', and 'type'.")

# --- Predict individual email
st.header("✉️ Try Your Own Email")

if model and vectorizer:
    user_input = st.text_area("Write an email here:")

    if st.button("Classify"):
        if not user_input.strip():
            st.warning("Please enter an email first.")
        else:
            cleaned = clean_text(user_input)
            vector = vectorizer.transform([cleaned])
            pred = model.predict(vector)[0]
            prob = model.predict_proba(vector)[0]

            st.subheader("Result:")
            st.markdown("🚨 **Spam**" if pred == 1 else "✅ **Not Spam**")
            st.write(f"Confidence: `{round(np.max(prob) * 100, 2)}%`")
else:
    st.info("Upload and train the model first to use prediction.")
