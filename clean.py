import pandas as pd
import numpy as np

print("="*60)
print("🔧 DATASET CLEANING & FIXING")
print("="*60)

# === Load dataset ===
df = pd.read_csv("spam.csv")
print(f"\n📊 Original dataset: {len(df)} rows")

# === Issue 1: Drop unnecessary columns ===
print("\n🗑️ Step 1: Removing unnecessary columns...")
# Keep only v1 (label) and cleaned_text
columns_to_keep = ['v1', 'cleaned_text']
df = df[columns_to_keep]
print(f"✅ Kept only: {columns_to_keep}")

# === Issue 2: Handle missing values in cleaned_text ===
print("\n🔍 Step 2: Handling missing values...")
missing_before = df['cleaned_text'].isnull().sum()
print(f"   Missing cleaned_text before: {missing_before}")

# Drop rows with missing cleaned_text
df = df.dropna(subset=['cleaned_text'])
print(f"✅ Dropped {missing_before} rows with missing text")
print(f"   Remaining rows: {len(df)}")

# === Issue 3: Remove empty texts ===
print("\n🧹 Step 3: Removing empty texts...")
empty_before = (df['cleaned_text'].str.strip() == '').sum()
df = df[df['cleaned_text'].str.strip() != '']
print(f"✅ Removed {empty_before} empty texts")

# === Issue 4: Remove very short texts (< 3 characters) ===
print("\n📏 Step 4: Removing very short texts...")
very_short = (df['cleaned_text'].str.len() < 3).sum()
df = df[df['cleaned_text'].str.len() >= 3]
print(f"✅ Removed {very_short} texts with < 3 characters")

# === Issue 5: Remove duplicates (CRITICAL) ===
print("\n🔄 Step 5: Removing duplicate texts...")
duplicates_before = df['cleaned_text'].duplicated().sum()
df = df.drop_duplicates(subset=['cleaned_text'], keep='first')
print(f"✅ Removed {duplicates_before} duplicate texts (9.05% of data!)")
print(f"   Remaining rows: {len(df)}")

# === Issue 6: Handle class imbalance ===
print("\n⚖️ Step 6: Checking class distribution...")
label_counts = df['v1'].value_counts()
print(f"   Ham: {label_counts['ham']}")
print(f"   Spam: {label_counts['spam']}")
spam_ratio = label_counts['spam'] / len(df) * 100
print(f"   Spam ratio: {spam_ratio:.2f}%")

if spam_ratio < 20:
    print("⚠️ Class imbalance detected (spam < 20%)")
    print("   💡 Will use class_weight='balanced' in model training")

# === Final cleanup: Reset index ===
df = df.reset_index(drop=True)

# === Save cleaned dataset ===
output_file = "spam_cleaned.csv"
df.to_csv(output_file, index=False)
print(f"\n💾 Saved cleaned dataset to: {output_file}")

# === Final Statistics ===
print("\n" + "="*60)
print("📊 FINAL DATASET STATISTICS")
print("="*60)
print(f"✅ Total samples: {len(df)}")
print(f"🟢 Ham: {(df['v1'] == 'ham').sum()} ({(df['v1'] == 'ham').sum()/len(df)*100:.2f}%)")
print(f"🔴 Spam: {(df['v1'] == 'spam').sum()} ({(df['v1'] == 'spam').sum()/len(df)*100:.2f}%)")

df['text_length'] = df['cleaned_text'].str.len()
df['word_count'] = df['cleaned_text'].str.split().str.len()

print(f"\n📏 Text length: {df['text_length'].mean():.1f} chars (avg)")
print(f"📝 Word count: {df['word_count'].mean():.1f} words (avg)")

print("\n" + "="*60)
print("🎯 SAMPLE CLEANED DATA")
print("="*60)

# Show some examples
print("\n🔴 SPAM Examples:")
for i, text in enumerate(df[df['v1'] == 'spam']['cleaned_text'].head(3), 1):
    print(f"{i}. {text[:100]}...")

print("\n🟢 HAM Examples:")
for i, text in enumerate(df[df['v1'] == 'ham']['cleaned_text'].head(3), 1):
    print(f"{i}. {text[:100]}...")

print("\n" + "="*60)
print("✅ DATASET READY FOR MODEL TRAINING!")
print("="*60)
print(f"📂 Use this file: {output_file}")
print("🚀 Now run your model.py with this cleaned dataset")