import pandas as pd
from sqlalchemy import create_engine

# connect to the same SQLite file
engine = create_engine('sqlite:///data/sentiment.db')

# read the CSV into a DataFrame
df = pd.read_csv(
    'data/training.1600000.processed.noemoticon.csv',
    encoding='latin-1',
    names=['label','tweet_id','date','query','user','text']
)

# keep only the columns we care about
df = df[['tweet_id','label','text']]

# append into the raw_tweets table
df.to_sql('raw_tweets', engine, if_exists='append', index=False)
