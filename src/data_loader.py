import pandas as pd

from src import config


def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(config.RAW_DATA_PATH)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
