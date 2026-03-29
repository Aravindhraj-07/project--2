from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import joblib

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# -------------------------------
# Load Pipeline Model (Vectorizer + Logistic)
# -------------------------------
try:
    model = joblib.load('fake_news_detector.pkl')
except Exception as e:
    print("Error loading model:", e)
    model = None


# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect_news():
    try:
        news_text = request.form['news_text']

        if not news_text.strip():
            flash('Please enter some news text!', 'error')
            return redirect(url_for('index'))

        if model is None:
            flash('Model not loaded properly!', 'error')
            return redirect(url_for('index'))

        prediction = predict_news(news_text)
        confidence = get_confidence(news_text)

        return render_template('result.html',
                               prediction=prediction,
                               confidence=confidence,
                               news_text=news_text)

    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('index'))


# -------------------------------
# Prediction Function
# -------------------------------
def predict_news(text):
    pred = model.predict([text])[0]
    return 'REAL' if pred == 1 else 'FAKE'


# -------------------------------
# Confidence Function
# -------------------------------
def get_confidence(text):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba([text])[0]
        return max(prob)
    return 0.0


# -------------------------------
# API Route
# -------------------------------
@app.route('/api/detect', methods=['POST'])
def api_detect():
    data = request.json
    text = data.get('text', '')

    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400

    prediction = predict_news(text)
    confidence = get_confidence(text)

    return jsonify({
        'prediction': prediction,
        'confidence': confidence,
        'message': f'This news is {prediction.lower()} with {confidence*100:.1f}% confidence'
    })


# -------------------------------
# Run App
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)