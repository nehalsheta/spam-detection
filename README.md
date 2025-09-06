📧 Email Spam Classifier (Streamlit App)

A simple Streamlit web application that allows you to upload an email dataset, train a spam classifier, and test your own email messages for spam detection. The model uses TF-IDF vectorization and a Naive Bayes classifier, and handles class imbalance using SMOTE.

🚀 Features

Upload your own dataset (CSV with title, text, and type columns)

Automatically cleans and preprocesses text

Handles class imbalance using SMOTE

Trains a Naive Bayes model with TF-IDF features

Saves the trained model and vectorizer using joblib

Allows live testing of custom email text

📁 Dataset Format

Your dataset must be a CSV file containing the following columns:

title	text	type
Subject of the email	Body of the email	spam or not spam

Example:

title,text,type
"Win a free iPhone","Click here to claim your prize!",spam
"Meeting Reminder","Don't forget the meeting at 10am",not spam

🧼 Text Preprocessing

The app performs the following text cleaning:

Converts to lowercase

Removes URLs

Removes mentions and hashtags

Removes special characters

🧠 Model Details

Vectorization: TfidfVectorizer with max_features=3000

Classifier: Multinomial Naive Bayes

Imbalance Handling: SMOTE from imblearn

Model Persistence: Saves and loads model/vectorizer using joblib

📦 Requirements

Install dependencies via pip:

pip install streamlit pandas scikit-learn imbalanced-learn joblib

▶️ How to Run

Run the Streamlit app:

streamlit run your_script_name.py


Replace your_script_name.py with the actual name of the Python file containing your code.

📊 After Training

Once the dataset is uploaded and the model is trained:

You’ll see accuracy and a classification report.

The trained model and vectorizer are saved as:

spam_model.pkl

vectorizer.pkl

📨 Try It Out

After training, use the text box to enter your own email content. The model will classify it as:

✅ Not Spam

🚨 Spam

With a confidence score!

📌 Notes

The model must be trained before you can use the prediction feature.

Make sure your dataset has the correct format; otherwise, an error will be shown.
