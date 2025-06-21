print("started")

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route('/')
def home():
    return "Welcome to the Fake News Detection API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # get the JSON input from the user
    if not data or 'news' not in data:
        return jsonify({'error': 'Please send a "text" field in JSON.'})

    input_text = data['news']
    transformed = vectorizer.transform([input_text])
    prediction = model.predict(transformed)[0]

    result = "Fake News" if prediction == 1 else "True News"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
