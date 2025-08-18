from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        message = request.form['message']
        if message.strip() == "":
            result = "⚠️ Please enter a message."
        else:
            # Preprocess and predict
            message_vector = vectorizer.transform([message])
            prediction = model.predict(message_vector)[0]
            result = "🟢 This is a HAM message (Not Spam)." if prediction == 0 else "🔴 This is a SPAM message!"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
