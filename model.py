import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Load cleaned dataset ===
df = pd.read_csv(r"C:\Users\ashut\OneDrive\Desktop\Email Spam\spam_cleaned.csv")

# === Drop rows with missing or empty cleaned_text ===
df = df.dropna(subset=['cleaned_text'])
df = df[df['cleaned_text'].str.strip() != '']

# === Map labels (ham -> 0, spam -> 1) ===
df['label'] = df['v1'].map({'ham': 0, 'spam': 1})

# === Features and target ===
X = df['cleaned_text']
y = df['label']

# === TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(max_features=3000)
X_vectorized = vectorizer.fit_transform(X)

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# === Train Naive Bayes model ===
model = MultinomialNB()
model.fit(X_train, y_train)

# === Predict on test set ===
y_pred = model.predict(X_test)

# === Evaluation ===
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Predicted Ham', 'Predicted Spam'],
    yticklabels=['Actual Ham', 'Actual Spam']
)
plt.title("📌 Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
import joblib

# Save model and vectorizer
joblib.dump(model, r"C:\Users\ashut\OneDrive\Desktop\Email Spam\spam_model.pkl")
joblib.dump(vectorizer, r"C:\Users\ashut\OneDrive\Desktop\Email Spam\tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved successfully.")

