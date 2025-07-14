import re
import sqlite3
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# 1) Connect to SQLite
conn = sqlite3.connect('data/sentiment.db')

# 2) Grab a small random sample of 100k raw tweets (text + label)
df = pd.read_sql(
    '''
    SELECT text, label
    FROM raw_tweets
    ORDER BY RANDOM()
    LIMIT 100000
    ''',
    conn
)

# 3) Basic cleaning
def clean(s):
    s = s.lower()
    s = re.sub(r'http\S+|@\S+|#\S+', '', s)
    s = re.sub(r'[^a-z\s]', '', s)
    return s

df['clean'] = df['text'].apply(clean)

# 4) Vectorize to TF‑IDF (max 1000 features)
vec = TfidfVectorizer(max_features=1000)
X = vec.fit_transform(df['clean'])

# 5) Prepare labels (0 stays 0; 4 → 1)
y = df['label'].map({0: 0, 4: 1}).values

# 6) Split and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# 7) Evaluate and print
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
