import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("spam_data.csv")
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

# Mapping kelas
class_map = {i: label for i, label in enumerate(le.classes_)}

# Gabungkan hasil ke dataframe
result_df = data.copy()
result_df['predicted'] = [class_map[p] for p in predictions]

for i, label in class_map.items():
    result_df[f'prob_{label}'] = probas[:, i]

# Tampilkan hasil
st.subheader("Prediction Results")
st.dataframe(result_df)

# Ringkasan akurasi
accuracy = (result_df['label'] == result_df['predicted']).mean()
st.subheader("Model Accuracy on CSV")
st.write(f"Accuracy: {accuracy:.2%}")
