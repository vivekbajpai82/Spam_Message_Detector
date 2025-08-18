import pandas as pd
import re
import string
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# === Load the CSV file ===
df = pd.read_csv(r"C:\Users\ashut\OneDrive\Desktop\Email Spam\spam.csv")

# === Normalize column names ===
df.columns = [col.lower().strip() for col in df.columns]

# === Auto-detect text column ===
text_col = 'text' if 'text' in df.columns else df.columns[1]

# === Remove empty or missing texts ===
df = df.dropna(subset=[text_col])
df = df[df[text_col].str.strip() != '']

# === Set of stopwords ===
stop_words = set(stopwords.words('english'))

# === Cleaning function ===
def clean_text(text, remove_stopwords=True):
    try:
        text = BeautifulSoup(text, "html.parser").get_text()
    except:
        pass
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)  # remove emails
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # remove mentions and hashtags
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # normalize whitespace
    if remove_stopwords:
        words = text.split()
        words = [word for word in words if word not in stop_words]
        text = ' '.join(words)
    return text

# === Apply cleaning to text column ===
df['cleaned_text'] = df[text_col].apply(lambda x: clean_text(str(x)))

# === Save the cleaned data ===
df.to_csv(r"C:\Users\ashut\OneDrive\Desktop\Email Spam\spam_cleaned.csv", index=False)

print("✅ Cleaning complete. File saved as 'spam_cleaned.csv'")
