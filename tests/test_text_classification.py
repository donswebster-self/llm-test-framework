# test_text_classification.py

import pytest
from transformers import pipeline

@pytest.fixture()
def sentiment_model():
    return pipeline("text-classification", model='finiteautomata/bertweet-base-sentiment-analysis')

def test_positive_sentiment(sentiment_model):
    """Test model classifies text with positive sentiment."""
    response = sentiment_model("Let's go to a fun concert this weekend.")

    assert response[0]["label"] == "POS"
    assert response[0]["score"] >= 0.5

def test_negative_sentiment(sentiment_model):
    """Test model classifies text with negative sentiment."""
    response = sentiment_model("The Great Depression of 1929 was painful for many.")

    assert response[0]["label"] == "NEG"
    assert response[0]["score"] >= 0.5

def test_neutral_sentiment(sentiment_model):
    """Test model classifies text with neutral sentiment."""
    response = sentiment_model("The table has four legs.")

    assert response[0]["label"] == "NEU"
    assert response[0]["score"] >= 0.5