import joblib
import pandas as pd

from src import config
from src.feature_engineering import add_engineered_features


def predict_file(input_csv_path: str) -> pd.DataFrame:
    model = joblib.load(config.MODEL_PATH)
    df = pd.read_csv(input_csv_path)
    model_input = add_engineered_features(df)
    preds = model.predict(model_input)

    result = df.copy()
    result["prediction"] = preds
    return result
