from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from utils import load_json, load_model


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates")
)


pipeline = load_model(MODELS_DIR / "preprocessing_pipeline.pkl")
schema = load_json(MODELS_DIR / "raw_schema.json")["raw_schema"]

churn_model = load_model(MODELS_DIR / "churn_classifier.pkl")
cluster_model = load_model(MODELS_DIR / "customer_clusters.pkl")
reg_model = load_model(MODELS_DIR / "monetary_regressor.pkl")


def prepare_input(form_dict: dict) -> np.ndarray:
    # 1) On construit un DF avec toutes les colonnes attendues
    df = pd.DataFrame([form_dict])

    # 2) Convertir en numérique quand possible (sinon reste object)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    # 3) Réindexer au schema attendu (colonnes manquantes => NaN)
    X_raw = df.reindex(columns=schema)

    # 4) Pipeline complet (FeatureEngineer + encoding + scaling + PCA)
    X_proc = pipeline.transform(X_raw)
    return X_proc


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict_classification", methods=["POST"])
def predict_classification_route():
    X_proc = prepare_input(request.form.to_dict())
    prob = float(churn_model.predict_proba(X_proc)[0, 1])
    label = "Risque de churn élevé" if prob >= 0.5 else "Client fidèle"
    return render_template("index.html", prediction_text=label, probability=f"{prob:.2%}")


@app.route("/predict_clustering", methods=["POST"])
def predict_clustering_route():
    X_proc = prepare_input(request.form.to_dict())
    cluster = int(cluster_model.predict(X_proc)[0])
    return render_template("index.html", prediction_text=f"Cluster {cluster}")


@app.route("/predict_regression", methods=["POST"])
def predict_regression_route():
    X_proc = prepare_input(request.form.to_dict())
    value = float(reg_model.predict(X_proc)[0])
    return render_template("index.html", prediction_text=f"Valeur prédite : {value:.2f}")


if __name__ == "__main__":
    app.run(debug=True)