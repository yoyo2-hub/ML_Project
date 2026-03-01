import os
import joblib
import pandas as pd
from flask import Flask, render_template, request
from src.utils import extract_ip_features

app = Flask(__name__)

# --- CONFIGURATION DES CHEMINS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models')

# --- CHARGEMENT DES OBJETS ---
model = joblib.load(os.path.join(MODEL_PATH, 'churn_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
imputer = joblib.load(os.path.join(MODEL_PATH, 'knn_imputer.pkl'))
model_columns = joblib.load(os.path.join(MODEL_PATH, 'model_columns.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Récupérer les données du formulaire
    raw_data = request.form.to_dict()
    df_input = pd.DataFrame([raw_data])

    # 2. Prétraitement identique au script preprocessing.py
    # Conversion numérique pour les champs du formulaire
    numeric_cols = ['Age', 'SatisfactionScore', 'CustomerTenureDays', 'FirstPurchaseDaysAgo', 'Frequency', 'MonetaryTotal']
    for col in numeric_cols:
        if col in df_input.columns:
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce')

    # Gestion de l'IP
    if 'LastLoginIP' in df_input.columns:
        ip_feats = extract_ip_features(df_input['LastLoginIP'].iloc[0])
        df_input['IP_Version'], df_input['IP_FirstOctet'], df_input['IP_IsPrivate'] = ip_feats
        df_input.drop(columns=['LastLoginIP'], inplace=True)

    # 3. Alignement des colonnes (One-Hot Encoding simulé)
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=model_columns, fill_value=0)

    # 4. Imputation et Scaling
    X_imputed = imputer.transform(df_input)
    X_scaled = scaler.transform(X_imputed)

    # 5. Prédiction
    prob = model.predict_proba(X_scaled)[0][1]
    prediction = "Risque de Churn élevé" if prob > 0.5 else "Client Fidèle"

    return render_template('index.html', 
                           prediction_text=prediction, 
                           probability=f"{prob:.2%}")

if __name__ == "__main__":
    app.run(debug=True)