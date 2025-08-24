import pandas as pd
import os

def load_data(raw_data_dir):
    """
    Load training, test, and sample submission datasets from the data/raw directory.

    Returns:
        train (pd.DataFrame): Training dataset
        test (pd.DataFrame): Test dataset
        sample_sub (pd.DataFrame): Sample submission dataset
    """
    #raw_data_dir = os.path.join("data", "raw")
    train_path = os.path.join(raw_data_dir, "train.csv")
    test_path = os.path.join(raw_data_dir, "test.csv")
    sample_sub_path = os.path.join(raw_data_dir, "SampleSubmission.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)

    return train, test, sample_sub


