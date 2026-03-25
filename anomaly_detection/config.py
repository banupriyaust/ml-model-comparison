"""
Centralized configuration for anomaly detection project.
"""

from pathlib import Path

# Paths
DB_PATH = Path(r"C:\Users\banup\Desktop\Masters thesis\chatbot\database\claims.db")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Data
SAMPLE_SIZE = 1_000_000
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42

# Model hyperparameters
ENCODING_DIM = 8
LATENT_DIM = 8
EPOCHS = 50
BATCH_SIZE = 2048
PATIENCE = 7

# Anomaly detection
DEFAULT_THRESHOLD_PERCENTILE = 97.5
SYNTHETIC_ANOMALY_FRACTION = 0.05
CONTAMINATION = 0.02

# Colors (matching existing project)
COLORS = ["#10a37f", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4"]
