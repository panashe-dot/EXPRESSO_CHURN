📞 Expresso Churn Prediction Challenge
Machine Learning Project for Customer Churn Prediction (Zindi Africa)
This repository contains a complete, modular implementation for the Expresso Churn Prediction Challenge, a supervised ML competition hosted on Zindi Africa. The objective is to predict whether a customer will churn—defined as becoming inactive and making no transactions for 90 days.
The project includes data ingestion, preprocessing, model training, prediction scripts, and an optional API layer for deployment.



📘 Overview of the Competition
Expresso is a telecom operator in Senegal and Mauritania, providing mobile data and airtime services.
The goal of the challenge is to build a machine learning model that predicts the probability that a customer will churn—stop making transactions for 90 consecutive days.
Models are evaluated using Log Loss, so calibrated, accurate probability outputs are essential.

🚀 Getting Started
1️⃣ Install Dependencies
Shellpip install -r requirements.txt``Show more lines
2️⃣ Download the Dataset
Download from the Zindi competition page and place all .csv files into:
src/data/raw/


🧹 Data Pipeline
Located in:
src/data/ingest_data.py
src/preprocess/
src/pipeline.py

Steps:

Ingest raw dataset
Clean and validate missing values
Encode categorical features
Engineered features such as:

Activity frequency
Revenue‑based metrics
Recharge behaviour
Usage ratios


Train‑validation splitting


🤖 Modeling
Main training logic is in:
src/models/trainer.py

Common model choices:

CatBoost (recommended for tabular data with mixed types)
XGBoost
LightGBM

The repository includes a pre-trained CatBoost model under:
src/api/Catboost.pkl


🔁 End‑to‑End Training
To run a full pipeline:
Shellpython src/pipeline.pyShow more lines
This executes:

Data ingestion
Preprocessing
Model training
Validation
Saving the final model to src/api/ or src/models/


🔮 Inference & Submission Generation
Use the predict.py script:
Shellpython src/predict.py \    --input src/data/raw/Test.csv \    --model src/api/Catboost.pkl \    --output src/data/predicted/submission.csvShow more lines
This outputs a submission-ready file for the Zindi platform.

🌐 Optional: API Deployment
The API module (FastAPI or Flask) lives in:
src/api/main.py

You can serve the model via:
Shelluvicorn src.api.main:app --reload``Show more lines

📓 Notebooks
The notebooks/ directory includes two Jupyter notebooks:


Exploration Notebook
Basic EDA, data visualization, churn behavior analysis.


Modeling Notebook
Iterative modeling experiments, hyperparameter tuning, feature engineering tests.


These notebooks were used to prototype the pipeline before converting it into modular Python scripts.

🧪 Evaluation Metric
The competition uses:
Log Loss

This requires probability predictions (not labels) and penalizes confident wrong predictions, making it ideal for churn forecasting.

🤝 Contributing
Contributions are welcome!
You can add:

Feature engineering improvements
New model architectures
Hyperparameter tuning experiments
API enhancements

