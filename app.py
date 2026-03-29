import streamlit as st
import joblib

# -------------------------------
# Load Model (Pipeline)
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load('fake_news_detector.pkl')

model = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("📰 Fake News Detection App")

st.write("Enter a news article below to check if it's real or fake.")

news_text = st.text_area("Enter News Text")

# -------------------------------
# Prediction
# -------------------------------
if st.button("Detect"):
    if news_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        prediction = model.predict([news_text])[0]

        if hasattr(model, "predict_proba"):
            confidence = max(model.predict_proba([news_text])[0])
        else:
            confidence = 0

        if prediction == 1:
            st.success(f"✅ This news is REAL")
        else:
            st.error(f"❌ This news is FAKE")

        st.write(f"Confidence: {confidence*100:.2f}%")