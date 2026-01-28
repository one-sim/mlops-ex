import pytest
from unittest import mock
from src.sentiment_model import analyze_sentiment, preprocess

def test_preprocess():
    text = "@user Check this out: http://example.com"
    result = preprocess(text)
    assert result == "@user Check this out: http"