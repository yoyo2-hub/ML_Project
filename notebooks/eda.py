import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

REPORTS_DIR = "reports"

def eda_report(df):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print("\n=== RAPPORT EDA ===")

    # Infos générales
    print(f"Taille du dataset : {df.shape}")
    print(df.info())
    print(df.describe())

    # Doublons
    duplicates = df.duplicated().sum()
    print(f"Doublons détectés : {duplicates}")

    # Valeurs manquantes
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print("\nColonnes avec valeurs manquantes :")
        print(missing)

    # Valeurs suspectes
    suspect_values = [-1, 999, 99, "Unknown", "unknown", "NA", ""]
    for col in df.select_dtypes(include=['object', 'number']).columns:
        for val in suspect_values:
            count = (df[col] == val).sum()
            if count > 0:
                print(f"⚠️ {col}: {count} valeurs suspectes '{val}'")

    # Histogrammes
    important_cols = ['Recency', 'Frequency', 'MonetaryTotal', 'Age']
    important_cols = [c for c in important_cols if c in df.columns]
    if important_cols:
        df[important_cols].hist(figsize=(12, 8), bins=30, color='skyblue', edgecolor='black')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "histograms.png"), dpi=300)
        plt.close()

    # Corrélations
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'CustomerID' in numeric_cols:
        numeric_cols.remove('CustomerID')
    if numeric_cols:
        corr_cols = numeric_cols[:10]  # limiter à 10 colonnes
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title("Corrélations")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "correlations.png"), dpi=300)
        plt.close()

    # Boxplots
    outlier_cols = ['MonetaryTotal', 'Age', 'Recency']
    outlier_cols = [c for c in outlier_cols if c in df.columns]
    if outlier_cols:
        fig, axes = plt.subplots(1, len(outlier_cols), figsize=(5*len(outlier_cols), 5))
        if len(outlier_cols) == 1:
            axes = [axes]
        for idx, col in enumerate(outlier_cols):
            sns.boxplot(y=df[col], ax=axes[idx], color='lightcoral')
            axes[idx].set_title(col)
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "boxplots.png"), dpi=300)
        plt.close()

    print("✅ Rapport EDA généré et graphiques sauvegardés dans /reports")

