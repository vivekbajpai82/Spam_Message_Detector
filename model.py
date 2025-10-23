import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import os

print("="*60)
print("🤖 ADVANCED SPAM CLASSIFIER TRAINING")
print("="*60)

# === Load CLEANED dataset ===
df = pd.read_csv("spam_cleaned_final.csv")
print(f"\n✅ Dataset loaded: {len(df)} samples")

# === Encode labels ===
df['label'] = df['v1'].map({'ham': 0, 'spam': 1})

ham_count = (df['label'] == 0).sum()
spam_count = (df['label'] == 1).sum()
print(f"🟢 Ham: {ham_count} ({ham_count/len(df)*100:.1f}%)")
print(f"🔴 Spam: {spam_count} ({spam_count/len(df)*100:.1f}%)")
print(f"⚠️ Class imbalance ratio: 1:{ham_count/spam_count:.1f}")

# === Split features & labels ===
X = df['cleaned_text']
y = df['label']

# === Enhanced TF-IDF Vectorization ===
print("\n🔤 Creating TF-IDF features with advanced settings...")
vectorizer = TfidfVectorizer(
    max_features=6000,           # More features for better learning
    ngram_range=(1, 3),          # Unigrams, Bigrams, Trigrams
    min_df=2,                    # Ignore words in < 2 docs
    max_df=0.9,                  # Ignore words in > 90% docs
    sublinear_tf=True,           
    strip_accents='unicode',
    lowercase=True,
    use_idf=True,
    smooth_idf=True,
    token_pattern=r'\b\w+\b'     # Better tokenization
)

X_vectorized = vectorizer.fit_transform(X)
print(f"✅ Vectorized shape: {X_vectorized.shape}")
print(f"   {X_vectorized.shape[0]} samples × {X_vectorized.shape[1]} features")

# === Train-test split with STRATIFICATION ===
print("\n📊 Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"✅ Train: {X_train.shape[0]} (Ham: {(y_train==0).sum()}, Spam: {(y_train==1).sum()})")
print(f"✅ Test: {X_test.shape[0]} (Ham: {(y_test==0).sum()}, Spam: {(y_test==1).sum()})")

# === Apply SMOTE for balancing ===
print("\n⚖️ Applying SMOTE for class balancing...")
print(f"   Before SMOTE - Spam: {(y_train==1).sum()}")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"   After SMOTE - Spam: {(y_train_balanced==1).sum()}")
print(f"✅ Training set balanced!")

# === Train MULTIPLE models ===
print("\n🎯 Training models...")

# Model 1: Naive Bayes
print("   1️⃣ Training Naive Bayes...")
nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train_balanced, y_train_balanced)

# Model 2: Logistic Regression  
print("   2️⃣ Training Logistic Regression...")
lr_model = LogisticRegression(
    C=10, 
    max_iter=1000, 
    random_state=42,
    class_weight='balanced'
)
lr_model.fit(X_train_balanced, y_train_balanced)

# === Ensemble: Use both models ===
print("   3️⃣ Creating Ensemble Model...")
ensemble = VotingClassifier(
    estimators=[('nb', nb_model), ('lr', lr_model)],
    voting='soft',
    weights=[1, 1.5]  # Give more weight to LR
)
ensemble.fit(X_train_balanced, y_train_balanced)

print("✅ All models trained!")

# === Evaluate ALL models ===
print("\n" + "="*60)
print("📊 MODEL COMPARISON")
print("="*60)

models = {
    'Naive Bayes': nb_model,
    'Logistic Regression': lr_model,
    'Ensemble': ensemble
}

best_model = None
best_f1 = 0
results = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fp': fp,
        'fn': fn
    }
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name
    
    print(f"\n📈 {name}:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   False Positives: {fp}")
    print(f"   False Negatives: {fn}")

print(f"\n🏆 BEST MODEL: {best_name}")

# === Detailed evaluation of best model ===
print("\n" + "="*60)
print(f"📊 DETAILED EVALUATION - {best_name}")
print("="*60)

y_pred = best_model.predict(X_test)
print("\n" + classification_report(y_test, y_pred, target_names=['Ham', 'Spam'], digits=4))

cm = confusion_matrix(y_test, y_pred)
print("📊 Confusion Matrix:")
print(f"                 Predicted")
print(f"                 Ham    Spam")
print(f"Actual  Ham    {cm[0][0]:>5}  {cm[0][1]:>5}")
print(f"        Spam   {cm[1][0]:>5}  {cm[1][1]:>5}")

# === Save best model ===
print("\n" + "="*60)
print("💾 SAVING BEST MODEL")
print("="*60)

os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

# Save metadata
metadata = {
    'model_type': best_name,
    'accuracy': results[best_name]['accuracy'],
    'f1_score': results[best_name]['f1'],
    'recall': results[best_name]['recall']
}
joblib.dump(metadata, "model/model_metadata.pkl")

print(f"✅ Best model saved: {best_name}")
print("✅ Files saved:")
print("   • model/spam_model.pkl")
print("   • model/tfidf_vectorizer.pkl")
print("   • model/model_metadata.pkl")

# === Test with sample messages ===
print("\n" + "="*60)
print("🧪 TESTING WITH REAL MESSAGES")
print("="*60)

test_messages = [
    "FREE! Win £1000 cash prize. Text WIN to 85555 now!",
    "Hey bro, you coming to the party tonight?",
    "URGENT: Your account has been compromised. Verify now!",
    "Thanks for the notes, really helpful for exam",
    "Congratulations! You've been selected for a free iPhone",
    "Can you pick me up at 5pm from station?",
    "WINNER! Claim your £500 prize by calling 09876",
    "Let's meet for coffee tomorrow afternoon",
    "Click here to get your free gift card worth £100",
    "Are you free this weekend?"
]

correct = 0
total = len(test_messages)

for i, msg in enumerate(test_messages, 1):
    msg_vec = vectorizer.transform([msg])
    prediction = best_model.predict(msg_vec)[0]
    proba = best_model.predict_proba(msg_vec)[0]
    
    # Determine if it looks like spam
    expected_spam = any(word in msg.lower() for word in ['free', 'win', 'prize', 'urgent', 'winner', 'claim', 'click'])
    
    if prediction == 1:
        label = "🔴 SPAM"
        confidence = proba[1] * 100
    else:
        label = "🟢 HAM"
        confidence = proba[0] * 100
    
    correct += (prediction == 1) == expected_spam
    
    print(f"\n{i}. {label} (confidence: {confidence:.1f}%)")
    print(f"   📝 {msg[:80]}{'...' if len(msg) > 80 else ''}")

print(f"\n✅ Visual accuracy: {correct}/{total} ({correct/total*100:.0f}%)")

# === Final Summary ===
print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)
print(f"🏆 Best Model: {best_name}")
print(f"📊 Test Accuracy: {results[best_name]['accuracy']*100:.2f}%")
print(f"🎯 Spam Detection Rate: {results[best_name]['recall']*100:.2f}%")
print(f"✓ Precision: {results[best_name]['precision']*100:.2f}%")
print(f"✓ F1-Score: {results[best_name]['f1']*100:.2f}%")
print(f"⚠️ False Positives: {results[best_name]['fp']}")
print(f"⚠️ False Negatives: {results[best_name]['fn']}")
print("\n🚀 Model is production-ready!")