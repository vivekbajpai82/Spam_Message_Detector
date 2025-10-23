import pandas as pd
import numpy as np
import re

# === Load dataset ===
print("="*60)
print("📂 LOADING DATASET")
print("="*60)

df = pd.read_csv("spam_cleaned.csv")
print(f"✅ Dataset loaded successfully!")
print(f"📊 Total rows: {len(df)}")
print(f"📋 Columns: {list(df.columns)}")

# === Basic Info ===
print("\n" + "="*60)
print("📊 DATASET INFO")
print("="*60)
print(df.info())

# === Check for missing values ===
print("\n" + "="*60)
print("🔍 MISSING VALUES CHECK")
print("="*60)
missing = df.isnull().sum()
print(missing)
if missing.sum() == 0:
    print("✅ No missing values found!")
else:
    print(f"⚠️ Total missing values: {missing.sum()}")

# === Check label distribution ===
print("\n" + "="*60)
print("📊 LABEL DISTRIBUTION")
print("="*60)
if 'v1' in df.columns:
    label_counts = df['v1'].value_counts()
    print(label_counts)
    print(f"\n📈 Percentage:")
    print(df['v1'].value_counts(normalize=True) * 100)
    
    # Check for class imbalance
    if len(label_counts) == 2:
        ratio = label_counts.min() / label_counts.max()
        if ratio < 0.3:
            print(f"\n⚠️ CLASS IMBALANCE DETECTED! Ratio: {ratio:.2f}")
            print("   Consider using techniques like SMOTE or class_weight='balanced'")
        else:
            print(f"\n✅ Good class balance! Ratio: {ratio:.2f}")

# === Sample data ===
print("\n" + "="*60)
print("👀 FIRST 5 ROWS")
print("="*60)
print(df.head())

# === Check cleaned_text column ===
if 'cleaned_text' in df.columns:
    print("\n" + "="*60)
    print("🧹 CLEANED TEXT ANALYSIS")
    print("="*60)
    
    # Check for empty strings
    empty_count = (df['cleaned_text'].str.strip() == '').sum()
    print(f"📝 Empty texts: {empty_count}")
    
    # Text length statistics
    df['text_length'] = df['cleaned_text'].str.len()
    print(f"\n📏 Text Length Statistics:")
    print(f"   Mean: {df['text_length'].mean():.2f} characters")
    print(f"   Median: {df['text_length'].median():.2f} characters")
    print(f"   Min: {df['text_length'].min()} characters")
    print(f"   Max: {df['text_length'].max()} characters")
    
    # Very short texts (potential issues)
    very_short = (df['text_length'] < 10).sum()
    print(f"\n⚠️ Very short texts (<10 chars): {very_short}")
    
    # Word count
    df['word_count'] = df['cleaned_text'].str.split().str.len()
    print(f"\n📝 Word Count Statistics:")
    print(f"   Mean: {df['word_count'].mean():.2f} words")
    print(f"   Median: {df['word_count'].median():.2f} words")
    print(f"   Min: {df['word_count'].min()} words")
    print(f"   Max: {df['word_count'].max()} words")
    
    # Check if text still has unwanted characters
    print("\n" + "="*60)
    print("🔍 CHARACTER ANALYSIS")
    print("="*60)
    
    has_numbers = df['cleaned_text'].str.contains(r'\d', na=False).sum()
    has_special_chars = df['cleaned_text'].str.contains(r'[^a-zA-Z\s]', na=False).sum()
    has_urls = df['cleaned_text'].str.contains(r'http|www', na=False, case=False).sum()
    has_emails = df['cleaned_text'].str.contains(r'\S+@\S+', na=False).sum()
    
    print(f"🔢 Texts with numbers: {has_numbers}")
    print(f"🔣 Texts with special characters: {has_special_chars}")
    print(f"🔗 Texts with URLs: {has_urls}")
    print(f"📧 Texts with emails: {has_emails}")
    
    # Sample cleaned texts
    print("\n" + "="*60)
    print("📖 SAMPLE CLEANED TEXTS (Random 5)")
    print("="*60)
    sample_indices = np.random.choice(len(df), min(5, len(df)), replace=False)
    for i, idx in enumerate(sample_indices, 1):
        label = df.iloc[idx]['v1'] if 'v1' in df.columns else 'Unknown'
        text = df.iloc[idx]['cleaned_text']
        print(f"\n{i}. [{label.upper()}]")
        print(f"   Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"   Length: {len(text)} chars, {len(text.split())} words")
    
    # Show some spam examples
    if 'v1' in df.columns and 'spam' in df['v1'].values:
        print("\n" + "="*60)
        print("🔴 SAMPLE SPAM MESSAGES (First 3)")
        print("="*60)
        spam_samples = df[df['v1'] == 'spam']['cleaned_text'].head(3)
        for i, text in enumerate(spam_samples, 1):
            print(f"\n{i}. {text[:150]}{'...' if len(text) > 150 else ''}")
    
    # Show some ham examples
    if 'v1' in df.columns and 'ham' in df['v1'].values:
        print("\n" + "="*60)
        print("🟢 SAMPLE HAM MESSAGES (First 3)")
        print("="*60)
        ham_samples = df[df['v1'] == 'ham']['cleaned_text'].head(3)
        for i, text in enumerate(ham_samples, 1):
            print(f"\n{i}. {text[:150]}{'...' if len(text) > 150 else ''}")

# === Duplicate check ===
print("\n" + "="*60)
print("🔄 DUPLICATE CHECK")
print("="*60)
if 'cleaned_text' in df.columns:
    duplicates = df['cleaned_text'].duplicated().sum()
    print(f"🔁 Duplicate texts: {duplicates}")
    if duplicates > 0:
        print(f"   ({duplicates/len(df)*100:.2f}% of dataset)")

# === Data Quality Score ===
print("\n" + "="*60)
print("🎯 DATA QUALITY SUMMARY")
print("="*60)

quality_issues = []
if missing.sum() > 0:
    quality_issues.append("Missing values detected")
if empty_count > 0:
    quality_issues.append(f"{empty_count} empty texts")
if very_short > len(df) * 0.1:
    quality_issues.append(f"Too many very short texts ({very_short})")
if duplicates > len(df) * 0.05:
    quality_issues.append(f"Too many duplicates ({duplicates})")

if not quality_issues:
    print("✅ Dataset looks GOOD!")
else:
    print("⚠️ Issues found:")
    for issue in quality_issues:
        print(f"   - {issue}")

print("\n" + "="*60)
print("💡 RECOMMENDATIONS")
print("="*60)

if has_urls > 0:
    print("📌 Consider removing URLs from text")
if has_numbers > len(df) * 0.5:
    print("📌 Many texts have numbers - decide if you want to keep them")
if very_short > 0:
    print("📌 Remove or investigate very short texts")
if duplicates > 0:
    print("📌 Remove duplicate entries: df.drop_duplicates(subset=['cleaned_text'], inplace=True)")

print("\n✅ Dataset check complete!")