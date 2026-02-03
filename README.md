# mlops-ex

Progetto di esempio per integrare pratiche MLOps in un sistema di analisi del sentiment: modello, monitoraggio continuo, rilevazione del drift e retraining.

Questo repository implementa:
- un wrapper per l'inferenza su un modello HuggingFace (in `src/sentiment_model.py`)
- un sistema di logging delle predizioni (`src/monitoring.py`)
- calcolo e persistenza delle metriche aggregate (`src/metrics.py`)
- rilevazione del drift confronto a una baseline (`src/drift_detection.py`)
- logica di valutazione e trigger per il retraining (`src/retraining.py`)
- notebook dimostrativo e di analisi: `monitoring_analysis.ipynb`
- test unitari in `tests/test_model.py`

Il progetto è pensato come base per dimostrare una pipeline MLOps completa per il monitoraggio in produzione.

## Struttura del repository

- `src/` - codice sorgente
	- `sentiment_model.py` - wrapper modello + preprocessing (con type hints)
	- `monitoring.py` - `PredictionLogger` (salva JSONL delle predizioni)
	- `metrics.py` - `MetricsTracker` per calcolare e persistere metriche
	- `drift_detection.py` - `DriftDetector` e reportistica
	- `retraining.py` - `RetrainingManager` e trigger
- `tests/` - test con `pytest`
- `monitoring_analysis.ipynb` - notebook che mostra download dataset, valutazione, logging, metriche, drift e retraining
- `requirements.txt` - dipendenze

## Requisiti e installazione

Questo progetto richiede Python 3.10+ (consigliato) e accesso a internet per scaricare il modello HuggingFace e i dataset.

Esempio rapido per l'ambiente virtuale e installazione:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Nota: la prima esecuzione scaricherà il modello `cardiffnlp/twitter-roberta-base-sentiment-latest` che può essere pesante.

## Come usare il codice

1) Eseguire il notebook dimostrativo (consigliato per esplorare le funzionalità):

```bash
pip install jupyterlab
jupyter lab monitoring_analysis.ipynb
```

2) Logging delle predizioni

Usa `PredictionLogger` per salvare ogni predizione in formato JSONL con i campi: `timestamp`, `text`, `sentiment`, `confidence`, `scores`.

Esempio d'uso (Python):

```python
from src.monitoring import PredictionLogger
from src.sentiment_model import analyze_sentiment

logger = PredictionLogger(log_dir='logs')
scores = analyze_sentiment('Questo prodotto è ottimo')
logger.log_prediction(text='Questo prodotto è ottimo', sentiment_scores=scores)
```

3) Calcolo metriche

Usa `MetricsTracker.calculate_metrics(logs)` passando i log caricati da `PredictionLogger.load_logs()` per ottenere:
- distribuzione per classe
- percentuali
- confidenza media e confidenza per classe

4) Drift detection

`DriftDetector` confronta la distribuzione corrente con una baseline salvata (`logs/baseline_distribution.json`) usando la distanza di Wasserstein e segnala drift se il valore supera `drift_threshold`.

Esempio:

```python
from src.drift_detection import DriftDetector
det = DriftDetector(baseline_file='logs/baseline.json', drift_threshold=0.15)
logs = logger.load_logs()
report = det.detect_drift(logs)
print(report.to_dict())
```

Se non esiste una baseline, `detect_drift` la creerà automaticamente con i dati forniti.

5) Trigger di retraining

`RetrainingManager.evaluate_retraining_need(logs)` valuta: numero minimo di nuovi campioni, confidenza media sotto soglia, e drift rilevato. Se le condizioni sono verificate, viene generato un trigger da salvare per azioni successive.

Esempio:

```python
from src.retraining import RetrainingManager
rm = RetrainingManager(min_samples_for_retraining=50, confidence_threshold=0.70)
trigger = rm.evaluate_retraining_need(logs)
print(trigger.to_dict())
```

## Eseguire i test

I test sono in `tests/test_model.py`. Per eseguirli:

```bash
pytest -q
```

Nota: alcuni test usano funzioni che caricano il modello o il dataset. La prima esecuzione può essere lenta.

## Configurazioni importanti

- `drift_threshold` (default 0.15): soglia per segnalare drift dalla distribuzione baseline.
- `min_samples_for_retraining` (default 50): numero minimo di nuove predizioni richieste per considerare un retraining.
- `confidence_threshold` (default 0.70): se la confidenza media scende sotto questa soglia, il retraining può essere considerato.

Questi valori sono configurabili nei costruttori di `DriftDetector` e `RetrainingManager`.

## Note operative

- Il notebook `monitoring_analysis.ipynb` mostra un flusso end-to-end: scarico dataset, eseguo valutazione sul test set (Accuracy/F1/Confusion Matrix), loggo predizioni campione, calcolo metriche, rilevo drift e valuto retraining.
