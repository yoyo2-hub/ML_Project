# 🛍️ Retail Analytics & Machine Learning Project

## 📌 Overview
This project applies machine learning techniques to analyze customer behavior in a retail context. It focuses on three main tasks: churn prediction (classification), monetary value estimation (regression), and customer segmentation (clustering). The solution is designed as an end-to-end pipeline, from data preprocessing to model deployment through a web application.

## 🎯 Objectives
The goal of this project is to predict customers who are likely to churn, estimate their future monetary value, and segment them into meaningful groups. These insights help improve customer retention strategies and support data-driven business decisions.

## 📊 Dataset
The dataset contains 4,372 customers and 52 features, including demographic, behavioral, and RFM (Recency, Frequency, Monetary) variables. Some missing and inconsistent values were identified and handled during preprocessing.

## ⚙️ Project Structure
project/  
├── data/ (raw, train_test, predictions)  
├── models/ (*.pkl, raw_schema.json)  
├── reports/ (plots, logs)  
├── src/ (preprocessing.py, train_model.py, evaluation.py, predict.py, features.py, utils.py)  
├── app.py  
└── README.md  

## 🔄 Machine Learning Pipeline
The pipeline includes data cleaning, feature engineering, encoding categorical variables, scaling numerical features, and applying PCA for dimensionality reduction. The processed data is then used to train classification, regression, and clustering models.

## 🤖 Models Used
- Churn Prediction: XGBoost Classifier  
- Monetary Prediction: Random Forest Regressor  
- Customer Segmentation: KMeans Clustering  

## 📈 Evaluation
Classification is evaluated using accuracy, precision, recall, and ROC AUC. Regression performance is measured using RMSE and R² score. Clustering is analyzed using the elbow method and PCA-based visualizations.

## 🚀 How to Run
1. Preprocessing: `python src/preprocessing.py`  
2. Train models: `python src/train_model.py`  
3. Run predictions: `python src/predict.py --mode test`  
4. Launch web app: `python app.py`  

## 🌐 Web Application
A Flask-based web application allows users to input customer data and receive real-time predictions for churn, monetary value, and customer segmentation.

## 📦 Outputs
The project generates processed datasets, trained models, prediction files, and visual reports for analysis and evaluation.

## ⚠️ Limitations
PCA reduces interpretability, KMeans assumes spherical clusters, and the classification threshold is fixed and may not be optimal for all business cases.

## 🔮 Future Improvements
Future work includes adding SHAP for explainability, handling class imbalance with SMOTE, improving clustering evaluation, and implementing model monitoring for data drift.

## 👩‍💻 Author
Chayma Dallel   

## 📅 Date
April 2026
