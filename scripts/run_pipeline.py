import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from scripts.download_dataset import download_kdd_dataset
from src.evaluate_model import evaluate_saved_model
from src.preprocess import preprocess_and_split
from src.train_model import train_and_save_model
from src.utils import ensure_project_dirs, setup_logging


def main():
    ensure_project_dirs()
    setup_logging()
    download_kdd_dataset(force=True)

    train_df, test_df = preprocess_and_split()
    model_summary = train_and_save_model(train_df)
    metrics = evaluate_saved_model(test_df)

    print("Pipeline completed.")
    print(f"Selected model: {model_summary['selected_model']}")
    print(f"CV F1 (weighted): {model_summary['selected_model_cv_f1_weighted']:.3f}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.3f}")


if __name__ == "__main__":
    main()
