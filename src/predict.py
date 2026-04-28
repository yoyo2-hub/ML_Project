from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils import check_files, load_json, load_model, save_data


DATA_DIR = Path("data/train_test")
RAW_PATH = Path("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
MODELS_DIR = Path("models")
OUT_DIR = Path("data/predictions")


def predict_on_test(output_path: Path) -> pd.DataFrame:
    check_files(
        [
            DATA_DIR / "X_test_pca.csv",
            DATA_DIR / "y_test.csv",
            DATA_DIR / "y_reg_test.csv",
            MODELS_DIR / "churn_classifier.pkl",
            MODELS_DIR / "monetary_regressor.pkl",
            MODELS_DIR / "customer_clusters.pkl",
        ]
    )

    X_test = pd.read_csv(DATA_DIR / "X_test_pca.csv")
    y_test_class = pd.read_csv(DATA_DIR / "y_test.csv")["Churn"].astype(int)
    y_test_reg = pd.read_csv(DATA_DIR / "y_reg_test.csv")["MonetaryTotal"].astype(float)

    churn_model = load_model(MODELS_DIR / "churn_classifier.pkl")
    reg_model = load_model(MODELS_DIR / "monetary_regressor.pkl")
    cluster_model = load_model(MODELS_DIR / "customer_clusters.pkl")

    churn_proba = churn_model.predict_proba(X_test)[:, 1]
    churn_pred = (churn_proba >= 0.5).astype(int)
    monetary_pred = reg_model.predict(X_test)
    cluster_pred = cluster_model.predict(X_test)

    acc = accuracy_score(y_test_class, churn_pred)
    prec = precision_score(y_test_class, churn_pred, zero_division=0)
    rec = recall_score(y_test_class, churn_pred, zero_division=0)
    f1 = f1_score(y_test_class, churn_pred, zero_division=0)

    print(f"TEST metrics: acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")
    print(f"Mean abs error monetary: {np.mean(np.abs(monetary_pred - y_test_reg)):.2f}")

    results = pd.DataFrame(
        {
            "idx": np.arange(len(X_test)),
            "churn_proba": churn_proba,
            "churn_pred": churn_pred,
            "churn_true": y_test_class.to_numpy(),
            "monetary_pred": monetary_pred,
            "monetary_true": y_test_reg.to_numpy(),
            "cluster": cluster_pred,
        }
    )
    save_data(results, output_path)
    return results


def predict_sample(n: int, output_path: Path) -> pd.DataFrame:
    check_files(
        [
            RAW_PATH,
            MODELS_DIR / "preprocessing_pipeline.pkl",
            MODELS_DIR / "raw_schema.json",
            MODELS_DIR / "churn_classifier.pkl",
            MODELS_DIR / "monetary_regressor.pkl",
            MODELS_DIR / "customer_clusters.pkl",
        ]
    )

    df = pd.read_csv(RAW_PATH).sample(n, random_state=42).reset_index(drop=True)

    # Build X_raw same schema as training input (before targets)
    schema = load_json(MODELS_DIR / "raw_schema.json")["raw_schema"]
    X_raw = df.reindex(columns=schema)

    pipeline = load_model(MODELS_DIR / "preprocessing_pipeline.pkl")
    churn_model = load_model(MODELS_DIR / "churn_classifier.pkl")
    reg_model = load_model(MODELS_DIR / "monetary_regressor.pkl")
    cluster_model = load_model(MODELS_DIR / "customer_clusters.pkl")

    X_proc = pipeline.transform(X_raw)

    churn_proba = churn_model.predict_proba(X_proc)[:, 1]
    churn_pred = (churn_proba >= 0.5).astype(int)
    monetary_pred = reg_model.predict(X_proc)
    cluster_pred = cluster_model.predict(X_proc)

    results = pd.DataFrame(
        {
            "idx": np.arange(len(X_raw)),
            "churn_proba": churn_proba,
            "churn_pred": churn_pred,
            "monetary_pred": monetary_pred,
            "cluster": cluster_pred,
        }
    )
    save_data(results, output_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "sample"], default="test")
    parser.add_argument("--output", default=str(OUT_DIR / "predictions.csv"))
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)

    if args.mode == "test":
        predict_on_test(output_path)
    else:
        predict_sample(args.n, output_path)


if __name__ == "__main__":
    main()