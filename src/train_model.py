import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train():
    print("--- DÉBUT DE L'ENTRAÎNEMENT ---")
    
    # 1. Chargement des données splittées (Sauvegardées par preprocessing.py)
    X_train = pd.read_csv('data/train_test/X_train.csv')
    y_train = pd.read_csv('data/train_test/y_train.csv').values.ravel()
    
    # 2. Initialisation du modèle (Paramètres optimaux trouvés via Optuna)
    model = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42)
    
    # 3. Entraînement
    print("Entraînement en cours...")
    model.fit(X_train, y_train)
    
    # 4. Sauvegarde
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/churn_model.pkl')
    print("✅ Modèle sauvegardé dans models/churn_model.pkl")

if __name__ == "__main__":
    train()