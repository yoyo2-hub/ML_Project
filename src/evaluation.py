from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from utils import save_model


DATA_DIR = Path("data/train_test")
MODELS_DIR = Path("models")


def main() -> None:
    X_train = pd.read_csv(DATA_DIR / "X_train_pca.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test_pca.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["Churn"].astype(int)
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["Churn"].astype(int)

    model = XGBClassifier(tree_method="hist", eval_metric="logloss", random_state=42)

    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=1,  # stable on Windows
        verbose=1,
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    proba = best.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    print("Best params:", grid.best_params_)
    print(f"Test AUC: {auc:.3f}")

    save_model(best, MODELS_DIR / "churn_classifier_optimized.pkl")


if __name__ == "__main__":
    main()