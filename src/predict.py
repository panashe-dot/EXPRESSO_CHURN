import joblib
import pandas as pd




test=pd.read_csv('data/processed/test.csv')
sample_sub = pd.read_csv('data/raw/SampleSubmission.csv')

model=joblib.load("models/Catboost.pkl")
predictions = model.predict_proba(test)[:, 1]
baseline_sub = sample_sub.copy()
baseline_sub['CHURN'] = predictions
baseline_sub.to_csv('data/predicted/catboost_expresso.csv', index=False)
print("Completed......................")

