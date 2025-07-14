from sqlalchemy import create_engine, Column, Integer, String, Text, MetaData, Table

# 1) point to a SQLite file in data/
engine = create_engine('sqlite:///data/sentiment.db')
meta = MetaData()

# 2) define raw_tweets table
raw = Table('raw_tweets', meta,
    Column('id', Integer, primary_key=True),
    Column('tweet_id', String),
    Column('label', Integer),
    Column('text', Text),
)

# 3) define features table
features = Table('features', meta,
    Column('id', Integer, primary_key=True),
    Column('tweet_id', String),
    Column('tfidf', String),      # JSON-encoded vector
    Column('pred_label', Integer),
)

# 4) create both tables in the database file
meta.create_all(engine)
