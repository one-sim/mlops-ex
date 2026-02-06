import pytest
from unittest import mock
from pathlib import Path
import tempfile
import json
from src.sentiment_model import analyze_sentiment, preprocess
from src.monitoring import PredictionLogger
from src.metrics import MetricsTracker
from src.drift_detection import DriftDetector


class TestPreprocess:
    """Test della funzione di preprocessing."""
    
    def test_preprocess_basic(self) -> None:
        """Test del preprocessing di base."""
        text: str = "@user Check this out: http://example.com"
        result: str = preprocess(text)
        assert result == "@user Check this out: http"
    
    def test_preprocess_with_mention(self) -> None:
        """Test del preprocessing con menzioni."""
        text: str = "@alice Hello @bob world"
        result: str = preprocess(text)
        assert "@user" in result
        assert "@alice" not in result
        assert "@bob" not in result
    
    def test_preprocess_with_url(self) -> None:
        """Test del preprocessing con URL."""
        text: str = "Check https://github.com and http://example.com"
        result: str = preprocess(text)
        assert "https://github.com" not in result
        assert "http://example.com" not in result
        assert result.count("http") == 2
    
    def test_preprocess_preserves_text(self) -> None:
        """Test che il preprocessing preserva il testo normale."""
        text: str = "This is a great product!"
        result: str = preprocess(text)
        assert "great" in result
        assert "product" in result


class TestAnalyzeSentiment:
    """Test della funzione di analisi del sentiment."""
    
    def test_analyze_sentiment_positive(self) -> None:
        """Test dell'analisi di testo positivo."""
        text: str = "I love this product! Excellent!"
        result = analyze_sentiment(text)
        
        assert isinstance(result, dict)
        assert len(result) == 3
        assert "Positivo" in result
        assert result["Positivo"] > result["Negativo"]
        assert 0 <= result["Positivo"] <= 1
    
    def test_analyze_sentiment_negative(self) -> None:
        """Test dell'analisi di testo negativo."""
        text: str = "This is terrible and awful"
        result = analyze_sentiment(text)
        
        assert "Negativo" in result
        assert result["Negativo"] > result["Positivo"]
    
    def test_analyze_sentiment_neutral(self) -> None:
        """Test dell'analisi di testo neutro."""
        text: str = "The weather is cloudy today"
        result = analyze_sentiment(text)
        
        assert "Neutro" in result
        # Per testo neutro, la classe neutro dovrebbe avere un punteggio buono
        assert isinstance(result["Neutro"], float)
    
    def test_analyze_sentiment_return_type(self) -> None:
        """Test del tipo di ritorno della funzione analyze_sentiment."""
        text: str = "Test text"
        result = analyze_sentiment(text)
        
        assert isinstance(result, dict)
        assert all(isinstance(v, float) for v in result.values())
        assert sum(result.values()) == pytest.approx(1.0, rel=1e-5)


class TestPredictionLogger:
    """Test del logger delle predizioni."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Crea una directory temporanea per i log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_logger_initialization(self, temp_log_dir: str) -> None:
        """Test dell'inizializzazione del logger."""
        logger = PredictionLogger(log_dir=temp_log_dir)
        assert Path(temp_log_dir).exists()
        assert logger.get_logs_count() == 0
    
    def test_log_prediction(self, temp_log_dir: str) -> None:
        """Test del logging di una predizione."""
        logger = PredictionLogger(log_dir=temp_log_dir)
        
        scores = {"Positivo": 0.8, "Neutro": 0.15, "Negativo": 0.05}
        log_entry = logger.log_prediction("Test text", scores)
        
        assert log_entry.sentiment == "Positivo"
        assert log_entry.confidence == 0.8
        assert logger.get_logs_count() == 1
    
    def test_load_logs(self, temp_log_dir: str) -> None:
        """Test del caricamento dei log."""
        logger = PredictionLogger(log_dir=temp_log_dir)
        
        # Logga alcune predizioni
        scores1 = {"Positivo": 0.7, "Neutro": 0.2, "Negativo": 0.1}
        scores2 = {"Negativo": 0.9, "Neutro": 0.05, "Positivo": 0.05}
        
        logger.log_prediction("Text 1", scores1)
        logger.log_prediction("Text 2", scores2)
        
        logs = logger.load_logs()
        assert len(logs) == 2
        assert logs[0].sentiment == "Positivo"
        assert logs[1].sentiment == "Negativo"
    
    def test_clear_logs(self, temp_log_dir: str) -> None:
        """Test della cancellazione dei log."""
        logger = PredictionLogger(log_dir=temp_log_dir)
        
        scores = {"Positivo": 0.8, "Neutro": 0.15, "Negativo": 0.05}
        logger.log_prediction("Test", scores)
        assert logger.get_logs_count() == 1
        
        logger.clear_logs()
        assert logger.get_logs_count() == 0


class TestMetricsTracker:
    """Test del tracker delle metriche."""
    
    @pytest.fixture
    def temp_metrics_dir(self):
        """Crea una directory temporanea per le metriche."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def sample_logs(self):
        """Crea log di esempio."""
        from src.monitoring import PredictionLog
        from datetime import datetime
        
        logs = [
            PredictionLog(
                timestamp=datetime.now().isoformat(),
                text="Great product!",
                sentiment="Positivo",
                confidence=0.85,
                scores={"Positivo": 0.85, "Neutro": 0.1, "Negativo": 0.05}
            ),
            PredictionLog(
                timestamp=datetime.now().isoformat(),
                text="Not good",
                sentiment="Negativo",
                confidence=0.75,
                scores={"Negativo": 0.75, "Neutro": 0.2, "Positivo": 0.05}
            ),
            PredictionLog(
                timestamp=datetime.now().isoformat(),
                text="Ok product",
                sentiment="Neutro",
                confidence=0.60,
                scores={"Neutro": 0.60, "Positivo": 0.25, "Negativo": 0.15}
            )
        ]
        return logs
    
    def test_calculate_metrics(self, temp_metrics_dir: str, sample_logs) -> None:
        """Test del calcolo delle metriche."""
        tracker = MetricsTracker(metrics_dir=temp_metrics_dir)
        metrics = tracker.calculate_metrics(sample_logs)
        
        assert metrics.total_predictions == 3
        assert len(metrics.sentiment_distribution) == 3
        assert metrics.sentiment_distribution["Positivo"] == 1
        assert metrics.sentiment_distribution["Negativo"] == 1
        assert metrics.sentiment_distribution["Neutro"] == 1
        assert pytest.approx(metrics.average_confidence, 0.01) == 0.733
    
    def test_save_and_load_metrics(self, temp_metrics_dir: str, sample_logs) -> None:
        """Test del salvataggio e caricamento delle metriche."""
        tracker = MetricsTracker(metrics_dir=temp_metrics_dir)
        metrics = tracker.calculate_metrics(sample_logs)
        tracker.save_metrics(metrics)
        
        history = tracker.get_metrics_history()
        assert len(history) == 1
        assert history[0].total_predictions == 3


class TestDriftDetector:
    """Test del detector di drift."""
    
    @pytest.fixture
    def temp_drift_dir(self):
        """Crea una directory temporanea per il drift."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def sample_logs(self):
        """Crea log di esempio."""
        from src.monitoring import PredictionLog
        from datetime import datetime
        
        logs = [
            PredictionLog(
                timestamp=datetime.now().isoformat(),
                text=f"Text {i}",
                sentiment="Positivo" if i % 2 == 0 else "Negativo",
                confidence=0.8,
                scores={"Positivo": 0.8 if i % 2 == 0 else 0.1,
                       "Negativo": 0.1 if i % 2 == 0 else 0.8,
                       "Neutro": 0.1}
            )
            for i in range(10)
        ]
        return logs
    
    def test_drift_detector_initialization(self, temp_drift_dir: str) -> None:
        """Test dell'inizializzazione del detector."""
        detector = DriftDetector(
            baseline_file=f"{temp_drift_dir}/baseline.json"
        )
        assert detector.get_baseline() is None
    
    def test_set_and_get_baseline(self, temp_drift_dir: str, sample_logs) -> None:
        """Test dell'impostazione e recupero del baseline."""
        detector = DriftDetector(
            baseline_file=f"{temp_drift_dir}/baseline.json",
            metrics_dir=temp_drift_dir
        )
        
        detector.set_baseline(sample_logs)
        baseline = detector.get_baseline()
        
        assert baseline is not None
        assert "distribution" in baseline
        assert "Positivo" in baseline["distribution"]
        assert "Negativo" in baseline["distribution"]
    
    def test_detect_drift(self, temp_drift_dir: str, sample_logs) -> None:
        """Test della rilevazione del drift."""
        detector = DriftDetector(
            baseline_file=f"{temp_drift_dir}/baseline.json",
            metrics_dir=temp_drift_dir,
            drift_threshold=0.3
        )
        
        # Imposta il baseline
        detector.set_baseline(sample_logs)
        
        # Testa la rilevazione del drift con gli stessi dati
        report = detector.detect_drift(sample_logs)
        
        assert report.drift_detected == False  # Stesso dataset
        assert report.drift_score >= 0.0
        assert report.drift_score <= 1.0