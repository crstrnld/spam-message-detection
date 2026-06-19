import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("spam_data.csv")

# Pastikan label string dan bersihkan kutip ganda
data['label'] = data['label'].astype(str).str.replace('"','').str.strip()

# Encode label
le = LabelEncoder()
y = le.fit_transform(data['label'])

# Vectorize text
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(data['text'])

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Streamlit UI
st.title("📩 Spam Detection from CSV")
st.write("Aplikasi ini membaca file spam_data.csv dan menampilkan hasil prediksi untuk semua pesan.")

# Prediksi semua baris di CSV
predictions = model.predict(X_vec)
probas = model.predict_proba(X_vec)

# Gabungkan hasil ke dataframe
result_df = data.copy()
result_df['predicted'] = [le.classes_[p] for p in predictions]
result_df['prob_spam'] = probas[:, list(le.classes_).index('spam')]
result_df['prob_ham'] = probas[:, list(le.classes_).index('ham')]

# Tampilkan hasil
st.subheader("Prediction Results")
st.dataframe(result_df)

# Ringkasan akurasi
accuracy = (result_df['label'] == result_df['predicted']).mean()
st.subheader("Model Accuracy on CSV")
st.write(f"Accuracy: {accuracy:.2%}")
