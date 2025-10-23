# app.py
from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# === Load model and vectorizer ===
model_path = os.path.join("model", "spam_model.pkl")
vectorizer_path = os.path.join("model", "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        message = request.form.get('message', '').strip()
        if not message:
            result = "⚠️ Please enter a message."
        else:
            # Vectorize & predict
            message_vector = vectorizer.transform([message])
            prediction = model.predict(message_vector)[0]
            result = (
                "🟢 This is a HAM message (Not Spam)." if prediction == 0
                else "🔴 This is a SPAM message!"
            )
    return render_template('index.html', result=result)


if __name__ == '__main__':
    # Production-safe entry point
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
