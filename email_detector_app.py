import streamlit as st
import joblib
import re

# Load vectorizer and model
vectorizer = joblib.load("Tfidf_vectorizer.pkl")
model = joblib.load("email_detection_model.pkl")

# Clean the email text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Streamlit UI
st.title("📧 Email Phishing & Spam Detector")
st.write("Paste your email content below to classify it.")

email_input = st.text_area("✉ Email Content", height=200)

if st.button("🔍 Check Email"):
    if email_input.strip() == "":
        st.warning("Please enter some email text.")
    else:
        cleaned = clean_text(email_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.error("⚠ This email is Phishing/Spam!")
        else:
            st.success("✅ This email is Legit.")

st.write("Made with ❤ by Ike-uchendu Joy .C.")