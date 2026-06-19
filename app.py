import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("spam_data.csv")  # kolom: 'text', 'label'
X = data['text']
y = data['label']

# Vectorize text
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Streamlit UI
st.title("📩 Spam Message Detector")
st.write("Masukkan pesan untuk cek apakah spam atau bukan:")

user_input = st.text_area("Message text")

if st.button("Predict"):
    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)[0]
    prediction_proba = model.predict_proba(input_vec)[0]

    st.subheader("Prediction")
    st.success("Spam" if prediction == 1 else "Not Spam")

    st.subheader("Prediction Probability")
    proba_df = pd.DataFrame([prediction_proba], columns=model.classes_)
    st.table(proba_df)
