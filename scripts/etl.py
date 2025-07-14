import re, json
import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer

# 1) Connect to SQLite database
engine = create_engine('sqlite:///data/sentiment.db')

# 2) Read raw tweets into a DataFrame
df = pd.read_sql('SELECT tweet_id, label, text FROM raw_tweets', engine)

# 3) Define a simple text-cleaning function
def clean_text(s):
    s = s.lower()
    s = re.sub(r'http\S+|@\S+|#\S+', '', s)      # strip URLs/mentions/hashtags
    s = re.sub(r'[^a-z\s]', '', s)               # keep only letters and spaces
    return s

# 4) Apply cleaning
df['clean'] = df['text'].apply(clean_text)

# 5) Vectorize with TF-IDF (limit features to speed things up)
vec = TfidfVectorizer(max_features=2000)
X = vec.fit_transform(df['clean'])
feature_names = vec.get_feature_names_out()

# 6) Prepare records for insertion
records = []
for tweet_id, row in zip(df['tweet_id'], X.toarray()):
    records.append({
        'tweet_id': tweet_id,
        'tfidf': json.dumps(row.tolist())
    })

feat_df = pd.DataFrame(records)

# 7) Write to the features table (it’ll create it if not exists)
feat_df.to_sql('features', engine, if_exists='replace', index=False)

print(f"ETL complete: {len(feat_df)} feature rows written.")
