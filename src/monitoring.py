"""
Modulo di monitoraggio per logging delle predizioni e gestione dei log.
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import os


@dataclass
class PredictionLog:
    """Classe per rappresentare un log di predizione."""
    timestamp: str
    text: str
    sentiment: str
    confidence: float
    scores: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte il log in dizionario."""
        return asdict(self)


class PredictionLogger:
    """Logger per le predizioni del modello di sentiment analysis."""
    
    def __init__(self, log_dir: str = "logs", log_file: str = "predictions.jsonl") -> None:
        """
        Inizializza il logger.
        
        Args:
            log_dir: Directory dove salvare i log
            log_file: Nome del file di log
        """
        self.log_dir: Path = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.log_file: Path = self.log_dir / log_file
        
        # Configura il logger di sistema
        self.logger: logging.Logger = logging.getLogger("sentiment_monitor")
        if not self.logger.handlers:
            handler: logging.FileHandler = logging.FileHandler(
                self.log_dir / "sentiment_model.log"
            )
            formatter: logging.Formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log_prediction(
        self,
        text: str,
        sentiment_scores: Dict[str, float]
    ) -> PredictionLog:
        """
        Logga una predizione.
        
        Args:
            text: Testo analizzato
            sentiment_scores: Dictionary con i punteggi dei sentiment
            
        Returns:
            PredictionLog con i dettagli della predizione
        """
        # Determina il sentiment principale e la confidenza
        sentiment: str = max(sentiment_scores, key=sentiment_scores.get)
        confidence: float = sentiment_scores[sentiment]
        
        # Crea il log
        log_entry: PredictionLog = PredictionLog(
            timestamp=datetime.now().isoformat(),
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            scores=sentiment_scores
        )
        
        # Salva nel file JSONL
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry.to_dict()) + '\n')
        
        self.logger.info(
            f"Predizione loggata: sentiment={sentiment}, "
            f"confidence={confidence:.4f}"
        )
        
        return log_entry
    
    def load_logs(self) -> list[PredictionLog]:
        """
        Carica tutti i log delle predizioni.
        
        Returns:
            Lista di PredictionLog
        """
        logs: list[PredictionLog] = []
        
        if not self.log_file.exists():
            return logs
        
        with open(self.log_file, 'r') as f:
            for line in f:
                if line.strip():
                    data: Dict[str, Any] = json.loads(line)
                    log: PredictionLog = PredictionLog(**data)
                    logs.append(log)
        
        return logs
    
    def clear_logs(self) -> None:
        """Cancella tutti i log delle predizioni."""
        if self.log_file.exists():
            self.log_file.unlink()
        self.logger.info("Log delle predizioni cancellato")
    
    def get_logs_count(self) -> int:
        """
        Ritorna il numero di predizioni loggare.
        
        Returns:
            Numero di log
        """
        return len(self.load_logs())
