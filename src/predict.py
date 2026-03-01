import joblib
import pandas as pd
import numpy as np
import os
from utils import extract_ip_features

def make_prediction(input_data):
    # 1. Charger les outils
    model = joblib.load('models/churn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    imputer = joblib.load('models/knn_imputer.pkl')
    model_columns = joblib.load('models/model_columns.pkl')

    # 2. Transformer l'entrée en DataFrame
    df = pd.DataFrame([input_data])

    # 3. Prétraitement (IP, Dates, etc.)
    if 'LastLoginIP' in df.columns:
        ip_feats = extract_ip_features(df['LastLoginIP'].iloc[0])
        df['IP_Version'], df['IP_FirstOctet'], df['IP_IsPrivate'] = ip_feats
        df.drop(columns=['LastLoginIP'], inplace=True)

    # 4. Alignement des colonnes (One-Hot Encoding)
    df = pd.get_dummies(df)
    df = df.reindex(columns=model_columns, fill_value=0)

    # 5. Imputation et Scaling
    df_imputed = imputer.transform(df)
    df_scaled = scaler.transform(df_imputed)

    # 6. Prédiction
    f_imputed = imputer.transform(df)
    df_scaled = pd.DataFrame(scaler.transform(f_imputed), columns=model_columns) # On remet les noms ici
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]
    return prediction, probability

if __name__ == "__main__":
    # Exemple de test rapide
    test_client = {
        'Age': 34,
        'SatisfactionScore': 2.5,
        'FirstPurchaseDaysAgo': 500,
        'Frequency': 2,
        'MonetaryTotal': 150,
        'LastLoginIP': '192.168.1.45'
    }
    
    res, prob = make_prediction(test_client)
    status = "CHURN" if res == 1 else "FIDÈLE"
    print(f"Résultat : {status} (Probabilité : {prob:.2%})")