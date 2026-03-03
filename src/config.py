from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "network_logs.csv"
TRAIN_PATH = DATA_DIR / "processed" / "train.csv"
TEST_PATH = DATA_DIR / "processed" / "test.csv"
SAMPLE_PATH = DATA_DIR / "sample" / "sample_logs.csv"

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
REPORTS_DIR = BASE_DIR / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
LOG_PATH = BASE_DIR / "logs" / "app.log"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "label"
KDD_SAMPLE_SIZE = 20000
CV_FOLDS = 5
