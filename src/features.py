from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Nettoyage + feature engineering, compatible scikit-learn.
    Doit recevoir un DataFrame pandas.
    """

    def __init__(self, date_col: str = "RegistrationDate"):
        self.date_col = date_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        df = X.copy()

        # --- Valeurs suspectes -> NaN
        suspect_values = {-1, 99, 999, "Unknown", "unknown", "NA", ""}
        for col in df.columns:
            df[col] = df[col].replace(list(suspect_values), np.nan)

        # --- Parsing date + variables calendrier
        if self.date_col in df.columns:
            reg_dt = pd.to_datetime(df[self.date_col], errors="coerce", dayfirst=True)
            df["RegYear"] = reg_dt.dt.year
            df["RegMonth"] = reg_dt.dt.month
        else:
            df["RegYear"] = np.nan
            df["RegMonth"] = np.nan

        # --- Features dérivées si colonnes présentes
        if {"MonetaryTotal", "Recency"}.issubset(df.columns):
            df["MonetaryPerDay"] = df["MonetaryTotal"] / (df["Recency"].astype(float) + 1.0)

        if {"MonetaryTotal", "Frequency"}.issubset(df.columns):
            freq = df["Frequency"].astype(float).replace(0, np.nan)
            df["AvgBasketValue"] = df["MonetaryTotal"] / freq

        if {"Recency", "CustomerTenure"}.issubset(df.columns):
            df["TenureRatio"] = df["Recency"] / (df["CustomerTenure"].astype(float) + 1.0)

        # --- Age cleaning
        if "Age" in df.columns:
            age = pd.to_numeric(df["Age"], errors="coerce")
            df.loc[~age.between(18, 81), "Age"] = np.nan

        # --- Drop colonnes inutiles / fuites (si présentes)
        drop_cols = [
            "Newsletter",
            "LastLoginIP",
            "RegistDate",  # au cas où
            "ChurnRiskCategory",
            "ChurnRisk",  # fuite potentielle si c'est une cible dérivée
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        return df