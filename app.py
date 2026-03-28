from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model
pipeline = joblib.load("fake_news_detector.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        news = request.form["news"]

        if news.strip() != "":
            result = pipeline.predict([news])[0]

            if result == 0:
                prediction = "Fake News ❌"
            else:
                prediction = "Real News ✅"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)