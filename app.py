import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Judul aplikasi
st.title("📩 Aplikasi Deteksi Pesan Spam Sederhana")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("spam_data.csv")

data = load_data()

st.subheader("Dataset")
st.write(data.head())

# Preprocessing
X = data['text']
y = data['label']

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Akurasi
accuracy = model.score(X_test, y_test)
st.write(f"📊 Akurasi model: {accuracy:.2f}")

# Input user
st.subheader("Coba Deteksi Pesan")
user_input = st.text_area("Masukkan teks pesan:")

if st.button("Deteksi"):
    if user_input.strip() != "":
        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)[0]
        st.success(f"Hasil deteksi: **{prediction.upper()}**")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu.")
