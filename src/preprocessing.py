import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import optuna
from utils import fill_age_from_cat, extract_ip_features # Import de tes fonctions

# --- 1. CHARGEMENT ---
input_path = 'data/raw/retail_customers_COMPLETE_CATEGORICAL.csv'
df = pd.read_csv(input_path)
df.columns = df.columns.str.strip()

# --- 2. NETTOYAGE & FEATURE ENGINEERING ---
age_map = {'18-24': 21, '25-34': 29, '35-44': 39, '45-54': 49, '55-64': 59, '65+': 72}
df['Age'] = df.apply(lambda r: fill_age_from_cat(r, age_map), axis=1)

# Création des nouveaux ratios métiers
df['AvgBasketValue'] = df['MonetaryTotal'] / (df['Frequency'] + 1)
df['TenureRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)

# Gestion IP
if 'LastLoginIP' in df.columns:
    df[['IP_Version', 'IP_FirstOctet', 'IP_IsPrivate']] = pd.DataFrame(
        df['LastLoginIP'].apply(extract_ip_features).tolist(), index=df.index)
    df.drop(columns=['LastLoginIP'], inplace=True)

# Nettoyage Support et Satisfaction
if 'SupportTicketsCount' in df.columns:
    df['SupportTicketsCount'] = df['SupportTicketsCount'].replace(-1, 0).clip(lower=0, upper=df['SupportTicketsCount'].quantile(0.99))

# --- 3. SAUVEGARDE DU FICHIER NETTOYÉ (L'étape qui manquait !) ---
# On crée le dossier s'il n'existe pas
os.makedirs('data/processed', exist_ok=True)

# On sauvegarde AVANT le get_dummies pour que le fichier reste lisible
df.to_csv('data/processed/base_donnees_nettoyee.csv', index=False, encoding='utf-8')
print("✅ Étape 1 : Fichier 'data/processed/base_donnees_nettoyee.csv' créé.")

# --- 4. PRÉPARATION MACHINE LEARNING ---
# Ajoute les segments suspects à la liste de suppression
cols_to_drop = [
    'CustomerID', 'ChurnRiskCategory', 'AccountStatus', 
    'RegistrationDate', 'NewsletterSubscribed',
    'CustomerType', 'RFMSegment', 'LoyaltyLevel',
    'Recency', 'TenureRatio' # On enlève les deux plus gros pour tester
]

# On s'assure que le drop est bien fait avant le get_dummies
df_ml = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Encodage One-Hot
df_final = pd.get_dummies(df_ml, drop_first=True)
print(df.groupby('Churn')['Recency'].mean())
# Split Stratifié
X = df_final.drop('Churn', axis=1)
y = df_final['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Imputation & Scaling
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Équilibrage SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_scaled, y_train)

# --- 5. OPTIMISATION & MODÈLE ---
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
    }
    clf = RandomForestClassifier(**param, random_state=42)
    clf.fit(X_res, y_res)
    return clf.score(X_test_scaled, y_test)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10) # n_trials réduit pour aller plus vite
# Après l'optimisation, réentraîne le meilleur modèle
best_model = RandomForestClassifier(**study.best_params, random_state=42)
best_model.fit(X_res, y_res)

# Affiche l'importance des variables
importances = pd.Series(best_model.feature_importances_, index=X.columns)
print("\n--- TOP 10 DES VARIABLES LES PLUS IMPORTANTES ---")
print(importances.sort_values(ascending=False).head(10))

print(f"✅ Étape 2 : Optimisation terminée. Meilleur score : {study.best_value:.4f}")