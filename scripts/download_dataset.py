import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_kddcup99

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src import config


def _decode_if_bytes(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    return value


def download_kdd_dataset(force: bool = True) -> Path:
    if config.RAW_DATA_PATH.exists() and not force:
        return config.RAW_DATA_PATH

    dataset = fetch_kddcup99(percent10=True)
    X = dataset.data
    y = dataset.target

    df = pd.DataFrame(
        {
            "duration": X[:, 0],
            "protocol": X[:, 1],
            "src_bytes": X[:, 4],
            "dst_bytes": X[:, 5],
            "flag": X[:, 3],
            "label": y,
        }
    )

    for col in df.columns:
        df[col] = df[col].map(_decode_if_bytes)

    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df["src_bytes"] = pd.to_numeric(df["src_bytes"], errors="coerce")
    df["dst_bytes"] = pd.to_numeric(df["dst_bytes"], errors="coerce")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].apply(
        lambda label: "benign" if label.startswith("normal") else "malicious"
    )

    if config.KDD_SAMPLE_SIZE and len(df) > config.KDD_SAMPLE_SIZE:
        df = df.sample(n=config.KDD_SAMPLE_SIZE, random_state=config.RANDOM_STATE)

    config.RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_DATA_PATH, index=False)
    return config.RAW_DATA_PATH


if __name__ == "__main__":
    path = download_kdd_dataset(force=True)
    print(f"Dataset saved to: {path}")
