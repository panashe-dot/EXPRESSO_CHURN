
# 📘 Overview of the Competition

Expresso is a telecom operator in **Senegal and Mauritania**, and the goal of this challenge is to build a machine learning model that predicts the **probability** that a customer will stop making transactions for **90 consecutive days** (i.e., churn).

---
# 🧪 Evaluation Metric

The competition uses **Log Loss**, which:
- Requires **probability predictions** between `0.0` and `1.0`
- Penalizes **confident but incorrect** predictions
- Encourages well‑calibrated models rather than hard classifications

---
# 🚀 Getting Started

## 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## 2️⃣ Download the Dataset  
Download the `.csv` files from the Zindi competition page and place them inside:
```
src/data/raw/
```

---
# 🧹 Data Pipeline

The data pipeline ensures a consistent, reproducible flow from raw files to model‑ready tensors.

Located in:
- `src/data/ingest_data.py`
- `src/preprocess/`
- `src/pipeline.py`

### Key Steps
#### Ingestion
Load raw files and validate structure.

#### Cleaning
Handle missing values, incorrect types, outliers.

#### Encoding
Prepare categorical features for gradient‑boosting models.

#### Feature Engineering
- Activity frequency
- Revenue-based metrics
- Recharge behaviour
- Usage ratios

#### Splitting
Stratified train-validation split.

---
# 🤖 Modeling

Main training logic:
```
src/models/trainer.py
```

Supported algorithms:
- CatBoost
- XGBoost
- LightGBM

Pre-trained model:
```
src/api/Catboost.pkl
```

---
# 🔁 End‑to‑End Training

```bash
python src/pipeline.py
```

---
# 🔮 Inference & Submission

```bash
python src/predict.py     --input src/data/raw/Test.csv     --model src/api/Catboost.pkl     --output src/data/predicted/submission.csv
```

---
# 🌐 API Deployment (Optional)

Serve the model using FastAPI:
```bash
uvicorn src.api.main:app --reload
```

---
# 📓 Notebooks

The `notebooks/` directory contains:
- Exploration notebook (EDA)
- Modeling notebook (feature engineering & tuning)

---
# 🤝 Contributing

You are welcome to:
- Improve feature engineering
- Add new models
- Enhance API endpoints

