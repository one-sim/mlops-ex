"""
Modulo per il tracking e calcolo delle metriche di monitoraggio.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import Counter
from .monitoring import PredictionLog


@dataclass
class SentimentMetrics:
    """Classe per rappresentare le metriche dei sentiment."""
    timestamp: str
    total_predictions: int
    sentiment_distribution: Dict[str, int]
    sentiment_percentages: Dict[str, float]
    average_confidence: float
    confidence_by_sentiment: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte le metriche in dizionario."""
        return asdict(self)


class MetricsTracker:
    """Tracker per il calcolo e monitoraggio delle metriche."""
    
    def __init__(self, metrics_dir: str = "logs", metrics_file: str = "metrics.jsonl") -> None:
        """
        Inizializza il tracker.
        
        Args:
            metrics_dir: Directory dove salvare le metriche
            metrics_file: Nome del file di metriche
        """
        self.metrics_dir: Path = Path(metrics_dir)
        self.metrics_dir.mkdir(exist_ok=True)
        
        self.metrics_file: Path = self.metrics_dir / metrics_file
    
    def calculate_metrics(self, logs: list[PredictionLog]) -> SentimentMetrics:
        """
        Calcola le metriche dai log delle predizioni.
        
        Args:
            logs: Lista di PredictionLog
            
        Returns:
            SentimentMetrics con le metriche calcolate
        """
        if not logs:
            return SentimentMetrics(
                timestamp=datetime.now().isoformat(),
                total_predictions=0,
                sentiment_distribution={},
                sentiment_percentages={},
                average_confidence=0.0,
                confidence_by_sentiment={}
            )
        
        # Distribuzione dei sentiment
        sentiments: list[str] = [log.sentiment for log in logs]
        sentiment_distribution: Dict[str, int] = dict(Counter(sentiments))
        
        total: int = len(logs)
        sentiment_percentages: Dict[str, float] = {
            sentiment: (count / total * 100)
            for sentiment, count in sentiment_distribution.items()
        }
        
        # Confidenza media
        average_confidence: float = sum(log.confidence for log in logs) / total
        
        # Confidenza media per sentiment
        confidence_by_sentiment: Dict[str, float] = {}
        for sentiment in sentiment_distribution.keys():
            sentiment_logs: list[PredictionLog] = [
                log for log in logs if log.sentiment == sentiment
            ]
            avg_conf: float = sum(log.confidence for log in sentiment_logs) / len(sentiment_logs)
            confidence_by_sentiment[sentiment] = avg_conf
        
        metrics: SentimentMetrics = SentimentMetrics(
            timestamp=datetime.now().isoformat(),
            total_predictions=total,
            sentiment_distribution=sentiment_distribution,
            sentiment_percentages=sentiment_percentages,
            average_confidence=average_confidence,
            confidence_by_sentiment=confidence_by_sentiment
        )
        
        return metrics
    
    def save_metrics(self, metrics: SentimentMetrics) -> None:
        """
        Salva le metriche nel file.
        
        Args:
            metrics: SentimentMetrics da salvare
        """
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics.to_dict()) + '\n')
    
    def get_metrics_history(self) -> list[SentimentMetrics]:
        """
        Ritorna la storia delle metriche.
        
        Returns:
            Lista di SentimentMetrics
        """
        metrics_history: list[SentimentMetrics] = []
        
        if not self.metrics_file.exists():
            return metrics_history
        
        with open(self.metrics_file, 'r') as f:
            for line in f:
                if line.strip():
                    data: Dict[str, Any] = json.loads(line)
                    metrics: SentimentMetrics = SentimentMetrics(**data)
                    metrics_history.append(metrics)
        
        return metrics_history
    
    def get_metrics_over_time(
        self,
        logs: list[PredictionLog],
        window_hours: int = 1
    ) -> list[Tuple[str, SentimentMetrics]]:
        """
        Calcola le metriche in finestre temporali.
        
        Args:
            logs: Lista di PredictionLog
            window_hours: Durata della finestra in ore
            
        Returns:
            Lista di tuple (timestamp, metriche) per ogni finestra
        """
        if not logs:
            return []
        
        # Ordina i log per timestamp
        sorted_logs: list[PredictionLog] = sorted(
            logs, key=lambda x: x.timestamp
        )
        
        # Crea finestre temporali
        metrics_over_time: list[Tuple[str, SentimentMetrics]] = []
        current_window: list[PredictionLog] = []
        
        start_time: datetime = datetime.fromisoformat(sorted_logs[0].timestamp)
        window_end: datetime = start_time + timedelta(hours=window_hours)
        
        for log in sorted_logs:
            log_time: datetime = datetime.fromisoformat(log.timestamp)
            
            if log_time > window_end:
                # Calcola metriche per la finestra corrente
                if current_window:
                    metrics: SentimentMetrics = self.calculate_metrics(current_window)
                    metrics_over_time.append((start_time.isoformat(), metrics))
                
                # Sposta alla finestra successiva
                start_time = window_end
                window_end = start_time + timedelta(hours=window_hours)
                current_window = []
            
            current_window.append(log)
        
        # Aggiungi l'ultima finestra
        if current_window:
            metrics = self.calculate_metrics(current_window)
            metrics_over_time.append((start_time.isoformat(), metrics))
        
        return metrics_over_time
    
    def clear_metrics(self) -> None:
        """Cancella tutte le metriche."""
        if self.metrics_file.exists():
            self.metrics_file.unlink()
