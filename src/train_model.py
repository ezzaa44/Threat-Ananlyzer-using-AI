import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.feature_engineering import add_engineered_features, build_preprocessor


def _build_model_candidates():
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=config.RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(random_state=config.RANDOM_STATE),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=config.RANDOM_STATE,
            solver="liblinear",
        ),
    }


def _make_pipeline(X_train: pd.DataFrame, classifier) -> Pipeline:
    preprocessor = build_preprocessor(X_train)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_and_save_model(train_df: pd.DataFrame) -> dict:
    X_train = train_df.drop(columns=[config.TARGET_COLUMN])
    y_train = train_df[config.TARGET_COLUMN]
    X_train = add_engineered_features(X_train)

    candidates = _build_model_candidates()
    comparison = {}
    best_name = None
    best_score = -1.0
    best_pipeline = None

    for name, classifier in candidates.items():
        pipeline = _make_pipeline(X_train, classifier)
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=config.CV_FOLDS,
            scoring="f1_weighted",
            n_jobs=1,
        )
        mean_score = float(scores.mean())
        comparison[name] = {
            "cv_f1_weighted_mean": mean_score,
            "cv_f1_weighted_scores": [float(s) for s in scores.tolist()],
        }

        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_pipeline = pipeline

    best_pipeline.fit(X_train, y_train)
    joblib.dump(best_pipeline, config.MODEL_PATH)

    summary = {
        "selected_model": best_name,
        "selected_model_cv_f1_weighted": best_score,
        "cv_results": comparison,
    }

    with open(config.MODEL_COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
