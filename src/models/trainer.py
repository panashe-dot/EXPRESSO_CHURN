import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_validate, learning_curve, RandomizedSearchCV,GridSearchCV, StratifiedKFold, cross_val_score, cross_val_predict,KFold
from sklearn.metrics import accuracy_score, precision_score,make_scorer, recall_score, f1_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import os
import joblib
import warnings
warnings.filterwarnings('ignore')
import warnings
warnings.filterwarnings("ignore", message="Dask dataframe query planning is disabled because dask-expr is not installed.")

def train_model(train,target):
    X_train, X_val, Y_train, Y_val = train_test_split(train, target, test_size=0.2, random_state=42)
   # Define models
    models = {
    "Logistic_Regression": LogisticRegression(),
    "Catboost": CatBoostClassifier(verbose=0),
    "Random_Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(verbosity=0, use_label_encoder=False),
    }


    for model_name, model in models.items():
        print(f"Evaluating model: {model_name}")
        model.fit(X_train, Y_train)
        Y_pred=model.predict(X_val)
        print("Accuracy:", accuracy_score(Y_val, Y_pred))
        print("Classification Report:\n", classification_report(Y_val, Y_pred))
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, f"models/{model_name}.pkl")


    return True
    
