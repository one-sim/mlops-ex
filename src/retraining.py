"""
Modulo per la gestione del retraining del modello.
"""
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from .monitoring import PredictionLog
from .metrics import MetricsTracker, SentimentMetrics
from .drift_detection import DriftDetector, DriftReport


@dataclass
class RetrainingTrigger:
    """Classe per rappresentare un trigger di retraining."""
    timestamp: str
    triggered: bool
    reason: str
    confidence_threshold_met: bool
    drift_threshold_met: bool
    min_samples_met: bool
    recommended_action: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte il trigger in dizionario."""
        return asdict(self)


class RetrainingManager:
    """Gestore del retraining del modello."""
    
    def __init__(
        self,
        min_samples_for_retraining: int = 100,
        confidence_threshold: float = 0.70,
        drift_detector: Optional[DriftDetector] = None,
        metrics_dir: str = "logs"
    ) -> None:
        """
        Inizializza il gestore di retraining.
        
        Args:
            min_samples_for_retraining: Numero minimo di campioni per il retraining
            confidence_threshold: Soglia di confidenza minima
            drift_detector: Istanza di DriftDetector
            metrics_dir: Directory dei log
        """
        self.min_samples_for_retraining: int = min_samples_for_retraining
        self.confidence_threshold: float = confidence_threshold
        
        self.drift_detector: DriftDetector = drift_detector or DriftDetector(metrics_dir=metrics_dir)
        self.metrics_tracker: MetricsTracker = MetricsTracker(metrics_dir)
        
        self.trigger_file: Path = Path(metrics_dir) / "retraining_triggers.jsonl"
        self.trigger_file.parent.mkdir(exist_ok=True)
    
    def evaluate_retraining_need(self, logs: list[PredictionLog]) -> RetrainingTrigger:
        """
        Valuta se è necessario il retraining del modello.
        
        Args:
            logs: Lista di PredictionLog recenti
            
        Returns:
            RetrainingTrigger con le indicazioni di retraining
        """
        timestamp: str = datetime.now().isoformat()
        
        # Verifica il numero di campioni
        min_samples_met: bool = len(logs) >= self.min_samples_for_retraining
        
        # Calcola le metriche
        metrics: SentimentMetrics = self.metrics_tracker.calculate_metrics(logs)
        
        # Verifica la confidenza media
        confidence_threshold_met: bool = metrics.average_confidence < self.confidence_threshold
        
        # Verifica il drift
        drift_report: DriftReport = self.drift_detector.detect_drift(logs)
        drift_threshold_met: bool = drift_report.drift_detected
        
        # Determina se il retraining è necessario
        triggered: bool = (
            min_samples_met and (confidence_threshold_met or drift_threshold_met)
        )
        
        # Genera il motivo e la raccomandazione
        reasons: list[str] = []
        
        if not min_samples_met:
            reasons.append(
                f"Campioni insufficienti: {len(logs)}/"
                f"{self.min_samples_for_retraining}"
            )
        
        if confidence_threshold_met:
            reasons.append(
                f"Confidenza media bassa: {metrics.average_confidence:.4f} "
                f"< {self.confidence_threshold}"
            )
        
        if drift_threshold_met:
            reasons.append(
                f"Drift rilevato: score={drift_report.drift_score:.4f}"
            )
        
        reason: str = " | ".join(reasons) if reasons else "Nessuna necessità di retraining"
        
        recommended_action: str = (
            "✅ ESEGUIRE RETRAINING" if triggered
            else "⏳ CONTINUE MONITORING"
        )
        
        trigger: RetrainingTrigger = RetrainingTrigger(
            timestamp=timestamp,
            triggered=triggered,
            reason=reason,
            confidence_threshold_met=confidence_threshold_met,
            drift_threshold_met=drift_threshold_met,
            min_samples_met=min_samples_met,
            recommended_action=recommended_action
        )
        
        return trigger
    
    def save_trigger(self, trigger: RetrainingTrigger) -> None:
        """
        Salva il trigger di retraining nel file.
        
        Args:
            trigger: RetrainingTrigger da salvare
        """
        with open(self.trigger_file, 'a') as f:
            f.write(json.dumps(trigger.to_dict()) + '\n')
    
    def get_trigger_history(self) -> list[RetrainingTrigger]:
        """
        Ritorna la storia dei trigger di retraining.
        
        Returns:
            Lista di RetrainingTrigger
        """
        triggers: list[RetrainingTrigger] = []
        
        if not self.trigger_file.exists():
            return triggers
        
        with open(self.trigger_file, 'r') as f:
            for line in f:
                if line.strip():
                    data: Dict[str, Any] = json.loads(line)
                    trigger: RetrainingTrigger = RetrainingTrigger(**data)
                    triggers.append(trigger)
        
        return triggers
    
    def get_last_triggered(self) -> Optional[RetrainingTrigger]:
        """
        Ritorna l'ultimo trigger di retraining che è stato attivato.
        
        Returns:
            RetrainingTrigger o None se nessun trigger è stato attivato
        """
        history: list[RetrainingTrigger] = self.get_trigger_history()
        triggered: list[RetrainingTrigger] = [t for t in history if t.triggered]
        
        return triggered[-1] if triggered else None
    
    def clear_triggers(self) -> None:
        """Cancella tutti i trigger di retraining."""
        if self.trigger_file.exists():
            self.trigger_file.unlink()
    
    def get_retraining_statistics(self) -> Dict[str, Any]:
        """
        Ritorna statistiche sul retraining.
        
        Returns:
            Dictionary con statistiche di retraining
        """
        history: list[RetrainingTrigger] = self.get_trigger_history()
        
        if not history:
            return {
                "total_evaluations": 0,
                "triggered_count": 0,
                "trigger_rate": 0.0,
                "last_triggered": None
            }
        
        triggered_count: int = sum(1 for t in history if t.triggered)
        
        return {
            "total_evaluations": len(history),
            "triggered_count": triggered_count,
            "trigger_rate": triggered_count / len(history) if history else 0.0,
            "last_triggered": history[-1].timestamp if history else None,
            "reasons": self._get_top_reasons(history)
        }
    
    @staticmethod
    def _get_top_reasons(triggers: list[RetrainingTrigger]) -> Dict[str, int]:
        """
        Estrae le ragioni più frequenti dai trigger.
        
        Args:
            triggers: Lista di RetrainingTrigger
            
        Returns:
            Dictionary con i conteggi delle ragioni
        """
        from collections import Counter
        
        reasons: list[str] = [t.reason for t in triggers if t.triggered]
        return dict(Counter(reasons).most_common(5))
