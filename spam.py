import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# ========================
# 1. Load the dataset
# ========================
df = pd.read_csv(r"C:\Users\ARABIA\OneDrive\Desktop\spamDetection\spamdetection\email_spam.csv")

# ========================
# 2. Combine 'title' and 'text' into one 'content' column
# ========================
df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# ========================
# 3. Label encoding
# ========================
df['label'] = df['type'].map({'spam': 1, 'not spam': 0})

# ========================
# 4. Clean the content
# ========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df['content'] = df['content'].apply(clean_text)

# ========================
# 5. Vectorization using TF-IDF
# ========================
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(df['content'])
y = df['label']

# ========================
# 6. Balance the dataset using SMOTE
# ========================
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# ========================
# 7. Train-test split
# ========================
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ========================
# 8. Train a Naive Bayes classifier
# ========================
model = MultinomialNB()
model.fit(X_train, y_train)

# ========================
# 9. Evaluate the model
# ========================
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ========================
# 10. Predict a custom email
# ========================
email_text = input("\nWrite your email content:\n")
email_text_cleaned = clean_text(email_text)
email_vector = vectorizer.transform([email_text_cleaned])
prediction = model.predict(email_vector)[0]
prob = model.predict_proba(email_vector)[0]

print("\nPrediction result:")
print("Spam" if prediction == 1 else "Not Spam")
print(f"Confidence: {round(np.max(prob) * 100, 2)}%")

# ========================
# 11. Save model and vectorizer
# ========================
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
