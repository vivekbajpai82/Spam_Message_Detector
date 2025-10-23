from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# === Load model and vectorizer ===
try:
    # Get the directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model", "spam_model.pkl")
    vectorizer_path = os.path.join(base_dir, "model", "tfidf_vectorizer.pkl")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("✅ Model and vectorizer loaded successfully.")

except FileNotFoundError:
    print("🛑 ERROR: Model or vectorizer file not found.")
    print("Please run the training script first to create the model files.")
    model = None
    vectorizer = None

@app.route('/', methods=['GET', 'POST'])
def index():
    result_data = None
    message = ""

    if request.method == 'POST':
        message = request.form.get('message', '').strip()
        
        if not message:
            result_data = {"error": "Please enter a message to classify."}
        elif model is None or vectorizer is None:
            result_data = {"error": "Model is not loaded. Please check server logs."}
        else:
            try:
                # Vectorize the message
                message_vector = vectorizer.transform([message])
                
                # Get probabilities (since we used 'soft' voting)
                probabilities = model.predict_proba(message_vector)[0]
                
                # Get the class with the highest probability
                prediction = probabilities.argmax()
                confidence = probabilities[prediction] * 100
                
                if prediction == 0:
                    label = "HAM (Not Spam)"
                    css_class = "ham"
                else:
                    label = "SPAM"
                    css_class = "spam"
                
                result_data = {
                    "label": label,
                    "confidence": f"{confidence:.2f}",
                    "css_class": css_class
                }

            except Exception as e:
                print(f"🛑 Error during prediction: {e}")
                result_data = {"error": "An error occurred during classification."}

    # Pass the original message back to the template
    return render_template('index.html', result_data=result_data, message=message)


if __name__ == '__main__':
    # Production-safe entry point
    port = int(os.environ.get("PORT", 5000))
    # Set debug=True for development, False for production
    app.run(host='0.0.0.0', port=port, debug=True)

