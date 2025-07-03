from datasets import load_dataset
import pandas as pd

# Load phishing email dataset from Hugging Face
dataset = load_dataset("zefang-liu/phishing-email-dataset", split="train")

# Convert to pandas DataFrame
df = pd.DataFrame(dataset)

# Save to CSV
df.to_csv("phishing_email_dataset.csv", index=False)

print("✅ Dataset saved as phishing_email_dataset.csv")

from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

#step 1: Load dataset
dataset = load_dataset("zefang-liu/phishing-email-dataset")
df = dataset['train'].to_pandas()

#step 2: Strip all column names to remove hidden spaces
df.columns = df.columns.str.strip()

# Rename to standard names
df = df.rename(columns={
    'Email Text': 'text',
    'Email Type': 'label'
})

# Drop index column if present
df = df.drop(columns=['unnamed: 0'], errors='ignore')
print(df.columns)  # Confirm new names

# Fix column names
df.columns = df.columns.str.strip()  # Remove spaces
df = df.rename(columns={'Email Text': 'text', 'Email Type': 'label'})

# Drop rows where 'text' is None or empty
df = df[df['text'].notnull()]           # Removes None
df = df[df['text'].str.strip() != '']   # Removes empty strings

#step 3: convert string labels to numeric
# Step 1: Normalize and convert string labels to integers
df['label'] = df['label'].str.strip().str.lower().map({
    'safe email': 0,
    'phishing email': 1
})
print(df['label'].value_counts())
# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Train model
model = LogisticRegression()
model.fit(X, df['label'])

# Save model and vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(model, "email_detection_model.pkl")

print("✅ Model retrained and saved successfully.")


