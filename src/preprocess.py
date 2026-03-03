import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.data_loader import load_raw_data, save_csv


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip().lower() for col in df.columns]

    if config.TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {config.TARGET_COLUMN}")

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("unknown")

    for col in df.select_dtypes(exclude=["object"]).columns:
        df[col] = df[col].fillna(df[col].median())

    return df


def preprocess_and_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_raw_data()
    df = clean_dataframe(df)

    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df[config.TARGET_COLUMN],
    )

    save_csv(train_df, str(config.TRAIN_PATH))
    save_csv(test_df, str(config.TEST_PATH))
    return train_df, test_df
