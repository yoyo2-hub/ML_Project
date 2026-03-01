import pandas as pd
import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import optuna
from utils import fill_age_from_cat, extract_ip_features

# --- CONFIGURATION DES CHEMINS DYNAMIQUES ---
# On définit la racine du projet par rapport à l'emplacement de ce script (src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'retail_customers_COMPLETE_CATEGORICAL.csv')

# --- 1. CHARGEMENT ---
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Fichier introuvable : {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)
df.columns = df.columns.str.strip()
print("✅ Fichier chargé avec succès !")

# --- 2. PARSING DE DATES ---
df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'], dayfirst=True, errors='coerce')
df['RegYear'] = df['RegistrationDate'].dt.year
df['RegMonth'] = df['RegistrationDate'].dt.month
df['RegYear'] = df['RegYear'].fillna(df['RegYear'].median())
df['RegMonth'] = df['RegMonth'].fillna(df['RegMonth'].median())

# --- 3. FEATURE ENGINEERING ---
age_map = {'18-24': 21, '25-34': 29, '35-44': 39, '45-54': 49, '55-64': 59, '65+': 72}
df['Age'] = df.apply(lambda r: fill_age_from_cat(r, age_map), axis=1)

df['AvgBasketValue'] = df['MonetaryTotal'] / (df['Frequency'] + 1)
df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)

if 'LastLoginIP' in df.columns:
    df[['IP_Version', 'IP_FirstOctet', 'IP_IsPrivate']] = pd.DataFrame(
        df['LastLoginIP'].apply(extract_ip_features).tolist(), index=df.index)

# --- 4. NETTOYAGE & SUPPRESSION ---
cols_to_drop = [
    'CustomerID', 'ChurnRiskCategory', 'AccountStatus', 
    'RegistrationDate', 'LastLoginIP', 'NewsletterSubscribed',
    'CustomerType', 'RFMSegment', 'LoyaltyLevel', 'Recency', 'TenureRatio'
]
df_ml = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# --- 5. VISUALISATION (Heatmap) ---
def generate_heatmap(df_input):
    df_numeric = df_input.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    os.makedirs(os.path.join(BASE_DIR, 'reports'), exist_ok=True)
    plt.title("Matrice de Corrélation des Features")
    plt.savefig(os.path.join(BASE_DIR, 'reports', 'heatmap_correlation.png'))
    print("✅ Heatmap sauvegardée dans reports/")

generate_heatmap(df_ml)

# --- 6. GESTION MULTICOLINÉARITÉ ---
corr_matrix = df_ml.select_dtypes(include=[np.number]).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.8)]
df_ml = df_ml.drop(columns=to_drop_corr)
print(f"🔥 Colonnes supprimées (Corrélation > 0.8) : {to_drop_corr}")

# --- 7. PRÉPARATION ML & SPLIT ---
df_final = pd.get_dummies(df_ml, drop_first=True)
X = df_final.drop('Churn', axis=1)
y = df_final['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Sauvegarde Split
split_dir = os.path.join(BASE_DIR, 'data', 'train_test')
os.makedirs(split_dir, exist_ok=True)
X_train.to_csv(os.path.join(split_dir, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(split_dir, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(split_dir, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(split_dir, 'y_test.csv'), index=False)

# --- 8. IMPUTATION, SCALING & SMOTE ---
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_scaled, y_train)

# --- 9. OPTUNA & EXPORT ---
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
    }
    clf = RandomForestClassifier(**param, random_state=42)
    clf.fit(X_res, y_res)
    return clf.score(X_test_scaled, y_test)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

best_model = RandomForestClassifier(**study.best_params, random_state=42)
best_model.fit(X_res, y_res)

# Sauvegarde .pkl
models_dir = os.path.join(BASE_DIR, 'models')
os.makedirs(models_dir, exist_ok=True)
joblib.dump(best_model, os.path.join(models_dir, 'churn_model.pkl'))
joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
joblib.dump(imputer, os.path.join(models_dir, 'knn_imputer.pkl'))
joblib.dump(list(X.columns), os.path.join(models_dir, 'model_columns.pkl'))

print("🚀 Pipeline terminé avec succès !")