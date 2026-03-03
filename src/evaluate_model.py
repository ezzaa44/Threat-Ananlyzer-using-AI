import json

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src import config
from src.feature_engineering import add_engineered_features


def evaluate_saved_model(test_df):
    model = joblib.load(config.MODEL_PATH)
    X_test = test_df.drop(columns=[config.TARGET_COLUMN])
    y_test = test_df[config.TARGET_COLUMN]
    X_test = add_engineered_features(X_test)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "classification_report": report,
    }

    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_test, y_pred, labels=["benign", "malicious"])
    metrics["confusion_matrix"] = cm.tolist()

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["benign", "malicious"],
            yticklabels=["benign", "malicious"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(config.CONFUSION_MATRIX_PATH)
        plt.close()
        metrics["confusion_matrix_image"] = str(config.CONFUSION_MATRIX_PATH)
    except ImportError:
        metrics["confusion_matrix_image"] = "skipped (install matplotlib and seaborn)"

    return metrics
