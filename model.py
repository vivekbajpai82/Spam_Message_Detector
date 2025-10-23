# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# === Load cleaned dataset ===
df = pd.read_csv("spam_cleaned.csv")

# === Drop missing or empty text ===
df = df.dropna(subset=['cleaned_text'])
df = df[df['cleaned_text'].str.strip() != '']

# === Encode labels ===
df['label'] = df['v1'].map({'ham': 0, 'spam': 1})

# === Split features & labels ===
X = df['cleaned_text']
y = df['label']

# === TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(max_features=3000)
X_vectorized = vectorizer.fit_transform(X)

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# === Train model ===
model = MultinomialNB()
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# === Save model & vectorizer ===
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved successfully in 'model' folder.")
