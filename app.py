import re

from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
import re
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load model
pipeline = joblib.load("fake_news_detector.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None


    if request.method == "POST":
        news = request.form["news"]
        

 

    if request.method == "POST":
        news = request.form["news"]

        if news.strip() != "":
            cleaned = clean_text(news)
            result = pipeline.predict([cleaned])[0]

            if result == 0:
                prediction = "Fake News ❌"
            else:
                prediction = "Real News ✅"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)