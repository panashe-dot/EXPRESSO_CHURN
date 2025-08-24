import pandas as pd
import numpy as np
from data.ingest_data import load_data


def preprocess(train,test):

    # splitting target variable and features
    test=test
    churn = train['CHURN']
    train = train.drop('CHURN', axis=1)
    data = pd.concat([train, test], sort=False)
    # Dropping zone columns
    data = data.drop(['ZONE1', 'ZONE2'], axis=1)

    # Fill missing values in categorical columns with "Missing_{column_name}"
    for col in data.select_dtypes(include=['object']).columns:
        data[col].fillna(f'Missing_{col}', inplace=True)
    # Fill missing values in numerical columns with the median
    for col in data.select_dtypes(include=[np.number]).columns:
        data[col].fillna(data[col].median(), inplace=True)


    data = data.drop(['MRG'], axis=1)
    data = data.drop(['REVENUE'], axis=1)
    data = pd.get_dummies(data, columns=['REGION', 'TOP_PACK', 'TENURE'], drop_first=True)
    data = data.drop('user_id', axis=1)

    train = data.iloc[:len(churn), :]
    test = data.iloc[len(churn):, :]



    return train,test,churn


