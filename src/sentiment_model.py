from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from scipy.special import softmax

# Carica modello e tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def preprocess(text):
    """Preprocess text (username and link placeholders for @/URLs)"""
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def analyze_sentiment(text):
    """Classify sentiment in positive, neutral, negative"""
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    output = model(**encoded_input)
    scores = softmax(output.logits.detach().numpy()[0])
    
    labels = ['Negativo', 'Neutro', 'Positivo']
    result = {label: float(score) for label, score in zip(labels, scores)}
    return result   