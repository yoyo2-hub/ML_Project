from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from xgboost import XGBClassifier

from utils import save_model


DATA_DIR = Path("data/train_test")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(DATA_DIR / "X_train_pca.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test_pca.csv")

    y_train_class = pd.read_csv(DATA_DIR / "y_train.csv")["Churn"].astype(int)
    y_test_class = pd.read_csv(DATA_DIR / "y_test.csv")["Churn"].astype(int)

    y_train_reg = pd.read_csv(DATA_DIR / "y_reg_train.csv")["MonetaryTotal"].astype(float)
    y_test_reg = pd.read_csv(DATA_DIR / "y_reg_test.csv")["MonetaryTotal"].astype(float)

    # -------------------------
    # 1) Classification (Churn)
    # -------------------------
    clf = XGBClassifier(
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
    )
    clf.fit(X_train, y_train_class)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test_class, y_pred)
    auc = roc_auc_score(y_test_class, y_proba)

    print(f"[Classification] Accuracy={acc:.3f} | AUC={auc:.3f}")
    save_model(clf, MODELS_DIR / "churn_classifier.pkl")

    # -------------------------
    # 2) Regression (Monetary)
    # -------------------------
    reg = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    reg.fit(X_train, y_train_reg)

    y_pred_reg = reg.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)))
    r2 = float(r2_score(y_test_reg, y_pred_reg))

    print(f"[Regression] RMSE={rmse:.2f} | R2={r2:.3f}")
    save_model(reg, MODELS_DIR / "monetary_regressor.pkl")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
    m1 = min(y_test_reg.min(), y_pred_reg.min())
    m2 = max(y_test_reg.max(), y_pred_reg.max())
    plt.plot([m1, m2], [m1, m2], "r--")
    plt.xlabel("Actual MonetaryTotal")
    plt.ylabel("Predicted MonetaryTotal")
    plt.title(f"Regression Actual vs Predicted (R2={r2:.3f})")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "regression_scatter.png", dpi=200)
    plt.close()

    # -------------------------
    # 3) Clustering (KMeans)
    # -------------------------
    inertias = []
    k_range = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_train)
        inertias.append(km.inertia_)

    # simple elbow: choose k at max second derivative
    dd = np.diff(inertias, n=2)
    optimal_k = list(k_range)[np.argmin(dd) + 1] if len(dd) > 0 else 4
    optimal_k = int(np.clip(optimal_k, 2, 10))
    print(f"[Clustering] optimal_k={optimal_k}")

    plt.figure(figsize=(7, 4))
    plt.plot(list(k_range), inertias, marker="o")
    plt.axvline(optimal_k, linestyle="--")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow curve")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "elbow_curve.png", dpi=200)
    plt.close()

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans.fit(X_train)
    save_model(kmeans, MODELS_DIR / "customer_clusters.pkl")

    clusters = kmeans.predict(X_train)
    plt.figure(figsize=(7, 5))
    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=clusters, cmap="viridis", s=20)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Clusters (k={optimal_k})")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "clusters_2d.png", dpi=200)
    plt.close()

    print("Training done. Models saved in /models")


if __name__ == "__main__":
    main()