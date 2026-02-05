"""
Modulo per la rilevazione del drift nel modello di sentiment analysis.
"""
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from scipy.stats import wasserstein_distance
import numpy as np
import numpy as np
from .monitoring import PredictionLog
from .metrics import MetricsTracker, SentimentMetrics


@dataclass
class DriftReport:
    """Classe per rappresentare un report di drift detection."""
    timestamp: str
    drift_detected: bool
    drift_score: float
    drift_threshold: float
    baseline_distribution: Dict[str, float]
    current_distribution: Dict[str, float]
    average_confidence_change: float
    recommendations: list[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte il report in dizionario."""
        return asdict(self)


class DriftDetector:
    """Rilevatore di drift per il modello di sentiment analysis."""
    
    def __init__(
        self,
        baseline_file: str = "logs/baseline_distribution.json",
        drift_threshold: float = 0.15,
        metrics_dir: str = "logs"
    ) -> None:
        """
        Inizializza il detector di drift.
        
        Args:
            baseline_file: File dove salvare la distribuzione baseline
            drift_threshold: Soglia per rilevare il drift (0-1)
            metrics_dir: Directory dei log
        """
        self.baseline_file: Path = Path(baseline_file)
        self.baseline_file.parent.mkdir(exist_ok=True)
        
        self.drift_threshold: float = drift_threshold
        self.metrics_tracker: MetricsTracker = MetricsTracker(metrics_dir)
        
        self.drift_report_file: Path = Path(metrics_dir) / "drift_reports.jsonl"
    
    def set_baseline(self, logs: list[PredictionLog]) -> None:
        """
        Imposta la distribuzione baseline dalle predizioni attuali.
        
        Args:
            logs: Lista di PredictionLog per creare il baseline
        """
        metrics: SentimentMetrics = self.metrics_tracker.calculate_metrics(logs)
        
        # Normalizza le percentuali a somma 1
        baseline_dist: Dict[str, float] = {
            sentiment: (pct / 100.0)
            for sentiment, pct in metrics.sentiment_percentages.items()
        }
        
        baseline_data: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "distribution": baseline_dist,
            "total_samples": metrics.total_predictions,
            "average_confidence": metrics.average_confidence
        }
        
        with open(self.baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
    
    def get_baseline(self) -> Optional[Dict[str, Any]]:
        """
        Carica la distribuzione baseline.
        
        Returns:
            Dictionary con il baseline o None se non esiste
        """
        if not self.baseline_file.exists():
            return None
        
        with open(self.baseline_file, 'r') as f:
            return json.load(f)
    
    def _calculate_wasserstein_distance(
        self,
        dist1: Dict[str, float],
        dist2: Dict[str, float]
    ) -> float:
        """
        Calcola la distanza di Wasserstein tra due distribuzioni.
        
        Args:
            dist1: Prima distribuzione
            dist2: Seconda distribuzione
            
        Returns:
            Distanza di Wasserstein normalizzata (0-1)
        """
        # Assicura che entrambe le distribuzioni abbiano le stesse chiavi
        all_keys: set[str] = set(dist1.keys()) | set(dist2.keys())
        
        values1: list[float] = [dist1.get(k, 0.0) for k in sorted(all_keys)]
        values2: list[float] = [dist2.get(k, 0.0) for k in sorted(all_keys)]
        
        # Calcola la distanza di Wasserstein
        distance: float = wasserstein_distance(values1, values2)
        
        # Normalizza a 0-1
        return min(distance, 1.0)
    
    def detect_drift(self, logs: list[PredictionLog]) -> DriftReport:
        """
        Rileva il drift confrontando le metriche attuali con il baseline.
        
        Args:
            logs: Lista di PredictionLog recenti
            
        Returns:
            DriftReport con i risultati della rilevazione
        """
        baseline: Optional[Dict[str, Any]] = self.get_baseline()
        
        if baseline is None:
            # Se non c'è baseline, imposta uno con i log attuali
            if logs:
                self.set_baseline(logs)
            return DriftReport(
                timestamp=datetime.now().isoformat(),
                drift_detected=False,
                drift_score=0.0,
                drift_threshold=self.drift_threshold,
                baseline_distribution={},
                current_distribution={},
                average_confidence_change=0.0,
                recommendations=["Baseline non trovato, creato con i dati attuali"]
            )
        
        # Calcola le metriche attuali
        current_metrics: SentimentMetrics = self.metrics_tracker.calculate_metrics(logs)
        current_dist: Dict[str, float] = {
            sentiment: (pct / 100.0)
            for sentiment, pct in current_metrics.sentiment_percentages.items()
        }
        
        baseline_dist: Dict[str, float] = baseline["distribution"]
        
        # Calcola il drift score usando la distanza di Wasserstein
        drift_score: float = self._calculate_wasserstein_distance(baseline_dist, current_dist)
        
        # Calcola il cambio nella confidenza media
        baseline_confidence: float = baseline.get("average_confidence", 0.5)
        confidence_change: float = current_metrics.average_confidence - baseline_confidence
        
        # Determina se c'è drift
        drift_detected: bool = drift_score > self.drift_threshold
        
        # Genera raccomandazioni
        recommendations: list[str] = []
        
        if drift_detected:
            recommendations.append(
                f"⚠️ DRIFT RILEVATO! Score: {drift_score:.4f} "
                f"(soglia: {self.drift_threshold})"
            )
            recommendations.append(
                "La distribuzione dei sentiment è cambiata significativamente"
            )
        
        if abs(confidence_change) > 0.05:
            direction: str = "diminuita" if confidence_change < 0 else "aumentata"
            recommendations.append(
                f"La confidenza media è {direction} di {abs(confidence_change):.4f}"
            )
        
        if confidence_change < -0.1:
            recommendations.append(
                "⚠️ CONSIDERARE IL RETRAINING: La confidenza è diminuita "
                "significativamente"
            )
        
        report: DriftReport = DriftReport(
            timestamp=datetime.now().isoformat(),
            drift_detected=drift_detected,
            drift_score=drift_score,
            drift_threshold=self.drift_threshold,
            baseline_distribution=baseline_dist,
            current_distribution=current_dist,
            average_confidence_change=confidence_change,
            recommendations=recommendations
        )
        
        return report
    
    def save_drift_report(self, report: DriftReport) -> None:
        """
        Salva il report di drift nel file.
        
        Args:
            report: DriftReport da salvare
        """
        def _to_serializable(o):
            if isinstance(o, dict):
                return {k: _to_serializable(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_to_serializable(v) for v in o]
            if isinstance(o, np.generic):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, datetime):
                return o.isoformat()
            return o

        serializable = _to_serializable(report.to_dict())
        with open(self.drift_report_file, 'a') as f:
            f.write(json.dumps(serializable) + '\n')
    
    def get_drift_history(self) -> list[DriftReport]:
        """
        Ritorna la storia dei report di drift.
        
        Returns:
            Lista di DriftReport
        """
        reports: list[DriftReport] = []
        
        if not self.drift_report_file.exists():
            return reports
        
        with open(self.drift_report_file, 'r') as f:
            for line in f:
                if line.strip():
                    data: Dict[str, Any] = json.loads(line)
                    report: DriftReport = DriftReport(**data)
                    reports.append(report)
        
        return reports
    
    def clear_drift_reports(self) -> None:
        """Cancella tutti i report di drift."""
        if self.drift_report_file.exists():
            self.drift_report_file.unlink()
