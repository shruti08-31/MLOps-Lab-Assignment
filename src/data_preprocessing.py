import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(data_path):
    """Load the dataset."""
    df = pd.read_csv(data_path)

    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y


def split_data(X, y, test_size, random_state):
    """Split data into training and testing sets."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )