import sys
import os
import pandas as pd
from data.ingest_data import load_data
from preprocess.preprocess import preprocess
from models.trainer import train_model

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))



def main():

    # Load data
    train, test, sample_sub = load_data('data/raw')

    # Preprocess data
    train_clean, test_clean, churn = preprocess(train,test)
    train_clean.to_csv('data/processed/train.csv', index=False)
    test_clean.to_csv('data/processed/test.csv', index=False)
    churn.to_csv('data/processed/churn.csv', index=False)

    # Train model
    print("Please wait for the pipeline to finish...")
    model = train_model(train_clean,churn)



    print("✅ Pipeline executed successfully")

if __name__ == "__main__":
    main()
