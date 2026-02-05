from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from scipy.special import softmax
from typing import Dict

# Local model path
model_path = "/workspaces/mlops-ex/twitter-roberta-base-sentiment"

# Load from local folder
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


def preprocess(text: str) -> str:
    """Preprocess text (username and link placeholders for @/URLs)
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Preprocessed text with placeholders
    """
    new_text: list[str] = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def analyze_sentiment(text: str) -> Dict[str, float]:
    """Classify sentiment in positive, neutral, negative
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with sentiment scores for each class
    """
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    output = model(**encoded_input)
    scores = softmax(output.logits.detach().numpy()[0])
    
    labels: list[str] = ['Negativo', 'Neutro', 'Positivo']
    result: Dict[str, float] = {label: float(score) for label, score in zip(labels, scores)}
    return result   