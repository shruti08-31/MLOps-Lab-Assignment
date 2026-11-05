
from src.data import load_data, split_data


def test_data_loading():
    X, y = load_data("data/raw/breast_cancer.csv")

    assert X.shape == (569, 30)
    assert len(y) == 569


def test_data_splitting():
    X, y = load_data("data/raw/breast_cancer.csv")

    X_train, X_test, y_train, y_test = split_data(
        X, y, 0.2, 42
    )

    assert len(X_train) == 455
    assert len(X_test) == 114
    assert len(y_train) == 455
    assert len(y_test) == 114
