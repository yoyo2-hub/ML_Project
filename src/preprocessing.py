from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import FeatureEngineer
from utils import log_report, save_data, save_json, save_model


RAW_PATH = Path("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
REPORTS_DIR = Path("reports")
DATA_OUT_DIR = Path("data/train_test")
MODELS_DIR = Path("models")

logging.basicConfig(
    filename=str(REPORTS_DIR / "preprocessing.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def eda_report(df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Missing values plot
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing) > 0:
        plt.figure(figsize=(10, 6))
        missing.head(30).plot(kind="bar")
        plt.title("Top missing ratios (30)")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "missing_ratios.png", dpi=200)
        plt.close()

    # Basic histograms
    important_cols = [c for c in ["Recency", "Frequency", "MonetaryTotal", "Age"] if c in df.columns]
    if important_cols:
        df[important_cols].hist(figsize=(12, 8), bins=30)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "histograms.png", dpi=200)
        plt.close()

    # Correlation (numeric)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 2:
        corr_cols = num_cols[:12]
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[corr_cols].corr(), cmap="coolwarm", center=0)
        plt.title("Correlation heatmap (subset)")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "correlations.png", dpi=200)
        plt.close()


def build_preprocessing_pipeline(n_components: int = 10) -> Pipeline:
    numeric_selector = make_column_selector(dtype_include=np.number)
    categorical_selector = make_column_selector(dtype_exclude=np.number)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_selector),
            ("cat", categorical_pipe, categorical_selector),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipe = Pipeline(
        steps=[
            ("fe", FeatureEngineer(date_col="RegistrationDate")),
            ("prep", preprocessor),
            ("pca", PCA(n_components=n_components, random_state=42)),
        ]
    )
    return pipe


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    # Reports
    eda_report(df)
    log_report(df, REPORTS_DIR / "data_report.txt")

    # Targets
    if "Churn" not in df.columns or "MonetaryTotal" not in df.columns:
        raise ValueError("Dataset must contain 'Churn' and 'MonetaryTotal' columns.")

    y_class = df["Churn"].astype(int)
    y_reg = df["MonetaryTotal"].astype(float)

    # X raw (keep original columns except targets + IDs)
    drop_cols = [c for c in ["Churn", "MonetaryTotal", "CustomerID"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    # Save raw schema for inference (Flask / predict_sample)
    save_json({"raw_schema": X.columns.tolist()}, MODELS_DIR / "raw_schema.json")

    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        X,
        y_class,
        y_reg,
        test_size=0.2,
        random_state=42,
        stratify=y_class,
    )

    # --- Optional: Outlier removal on TRAIN (numeric only, before fitting pipeline)
    # We do it on a *temporary* numeric-only view
    fe_tmp = FeatureEngineer(date_col="RegistrationDate")
    X_train_tmp = fe_tmp.transform(X_train)

    num_cols = X_train_tmp.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 0:
        iso = IsolationForest(contamination=0.06, random_state=42)
        X_iso = X_train_tmp[num_cols].copy()
        X_iso = X_iso.fillna(X_iso.median(numeric_only=True))
        mask = iso.fit_predict(X_iso) == 1

        X_train = X_train.loc[mask].reset_index(drop=True)
        y_train_class = y_train_class.loc[mask].reset_index(drop=True)
        y_train_reg = y_train_reg.loc[mask].reset_index(drop=True)

    # Build + fit pipeline
    pipe = build_preprocessing_pipeline(n_components=10)
    X_train_pca = pipe.fit_transform(X_train)
    X_test_pca = pipe.transform(X_test)

    # Save pipeline
    save_model(pipe, MODELS_DIR / "preprocessing_pipeline.pkl")

    # Save processed matrices
    pca_dim = X_train_pca.shape[1]
    cols = [f"PC{i+1}" for i in range(pca_dim)]

    save_data(pd.DataFrame(X_train_pca, columns=cols), DATA_OUT_DIR / "X_train_pca.csv")
    save_data(pd.DataFrame(X_test_pca, columns=cols), DATA_OUT_DIR / "X_test_pca.csv")

    save_data(pd.DataFrame({"Churn": y_train_class}), DATA_OUT_DIR / "y_train.csv")
    save_data(pd.DataFrame({"Churn": y_test_class}), DATA_OUT_DIR / "y_test.csv")

    save_data(pd.DataFrame({"MonetaryTotal": y_train_reg}), DATA_OUT_DIR / "y_reg_train.csv")
    save_data(pd.DataFrame({"MonetaryTotal": y_test_reg}), DATA_OUT_DIR / "y_reg_test.csv")

    print("Preprocessing done.")
    print(f"Saved: {DATA_OUT_DIR}/X_train_pca.csv, X_test_pca.csv and targets.")


if __name__ == "__main__":
    main()