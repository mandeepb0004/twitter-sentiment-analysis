# Twitter Sentiment Analysis

A lightweight end‑to‑end pipeline to classify tweet sentiment using the open Sentiment140 dataset. This project demonstrates database design, ETL workflows, and a simple machine‑learning model: we ingest labeled tweets into SQLite, clean and vectorize text with TF‑IDF, train a logistic regression classifier, and evaluate accuracy—all with reproducible scripts.

## Quickstart

1. **Clone the repository**  
   ```bash
   git clone https://github.com/mandeep0004/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```

2. **Set up Python environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download the Sentiment140 dataset**  
   - Visit http://help.sentiment140.com/for-students  
   - Download `training.1600000.processed.noemoticon.csv`  
   - Move it into the `data/` folder:  
     ```bash
     mv ~/Downloads/training.1600000.processed.noemoticon.csv data/
     ```

4. **Initialize the database**  
   ```bash
   python scripts/create_db.py
   python scripts/load_raw.py
   ```

5. **Run ETL and feature extraction**  
   ```bash
   python scripts/etl.py
   ```

6. **Train and evaluate the model**  
   ```bash
   python scripts/train_model.py > reports/model_report.txt
   ```

7. **View results**  
   - Open `reports/model_report.txt` for accuracy, precision, recall, and F1 score.  
   - Optionally, generate confusion‑matrix plots in `notebooks/metrics.ipynb`.

You’re all set—each script runs from raw data to evaluation. Explore the code in `scripts/` and inspect `data/sentiment.db` for stored tweets and features.