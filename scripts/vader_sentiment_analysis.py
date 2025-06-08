# scripts/vader_sentiment_analysis.py

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

logging.basicConfig(level=logging.INFO)

def load_reviews(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = ['review', 'rating', 'bank']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column in dataset: {col}")
    df.dropna(subset=["review"], inplace=True)
    return df


def apply_vader_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment_label(text: str) -> str:
        score = analyzer.polarity_scores(text)["compound"]
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    df["vader_score"] = df["review"].astype(str).apply(lambda x: analyzer.polarity_scores(x)["compound"])
    df["vader_label"] = df["review"].astype(str).apply(get_sentiment_label)
    return df


def aggregate_sentiment_by_bank_rating(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["bank", "rating"])["vader_label"].value_counts(normalize=True).unstack().fillna(0)
    return agg


def save_results(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    logging.info(f"Saved sentiment-labeled data to: {path}")
