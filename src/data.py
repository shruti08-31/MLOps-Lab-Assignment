
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_data(data_path, target="target"):
    if not Path(data_path).exists():
        raise FileNotFoundError("Dataset file not found")

    df = pd.read_csv(data_path)

    if target not in df.columns:
        raise ValueError("Target column not found")

    if df.empty:
        raise ValueError("Dataset is empty")

    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values")

    X = df.drop(columns=[target])
    y = df[target]

    if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns):
        raise ValueError("All input features must be numeric")

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
