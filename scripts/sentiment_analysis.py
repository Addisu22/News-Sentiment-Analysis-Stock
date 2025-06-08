# scripts/sentiment_analysis.py

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    required_cols = ['review', 'rating', 'bank']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must include columns: {required_cols}")
    df.dropna(subset=["review"], inplace=True)
    return df


def analyze_sentiment_vader(df: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    logging.info("Running VADER sentiment analysis...")
    df["compound"] = df["review"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])
    df["sentiment_label"] = df["compound"].apply(
        lambda score: "positive" if score > 0.05 else ("negative" if score < -0.05 else "neutral")
    )
    return df


def analyze_sentiment_bert(df: pd.DataFrame) -> pd.DataFrame:
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    logging.info("Running DistilBERT sentiment analysis (this may take time)...")
    results = sentiment_pipeline(df["review"].astype(str).tolist(), truncation=True, batch_size=8)
    df["bert_sentiment"] = [r["label"].lower() for r in results]
    df["bert_score"] = [r["score"] for r in results]
    return df


def aggregate_sentiment(df: pd.DataFrame, label_col: str = "sentiment_label") -> pd.DataFrame:
    logging.info("Aggregating sentiment by bank and rating...")
    return df.groupby(["bank", "rating"])[label_col].value_counts(normalize=True).unstack().fillna(0)


def save_output(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, index=False)
    logging.info(f"Saved results to {filepath}")
