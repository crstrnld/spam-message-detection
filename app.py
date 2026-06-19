import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("spam_data.csv")

# Bersihkan label dari kutip ganda
data['label'] = data['label'].str.replace('"', '').str.strip()

# Encode label ke angka
le = LabelEncoder()
y = le.fit_transform(data['label'])  # spam=1, ham=0

# Vectorize text
X = data['text']
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Streamlit UI
st.title("📩 Spam Message Detector")
st.write("Masukkan pesan untuk cek apakah spam atau bukan:")

user_input = st.text_area("Message text")

if st.button("Predict"):
    if user_input.strip() != "":
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        prediction_proba = model.predict_proba(input_vec)[0]

        # Hasil prediksi
        st.subheader("Prediction")
        st.success("Spam" if prediction == 1 else "Not Spam")

        # Tabel probabilitas
        st.subheader("Prediction Probability")
        proba_df = pd.DataFrame([prediction_proba], columns=le.classes_)
        st.table(proba_df)

        # Grafik bar chart
        st.subheader("Probability Visualization")
        fig, ax = plt.subplots()
        ax.bar(le.classes_, prediction_proba, color=['red', 'green'])
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
    else:
        st.warning("Masukkan teks pesan terlebih dahulu.")
