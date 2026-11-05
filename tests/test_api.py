
import pandas as pd
from fastapi.testclient import TestClient
from src.api import app, feature_names


client = TestClient(app)


def get_valid_features():
    df = pd.read_csv("data/raw/breast_cancer.csv")
    X = df.drop(columns=["target"])
    return X.iloc[0].to_dict()


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_valid_prediction():
    response = client.post(
        "/predict",
        json={"features": get_valid_features()}
    )

    assert response.status_code == 200
    assert response.json()["prediction"] in [0, 1]
    assert 0 <= response.json()["probability"] <= 1


def test_missing_feature():
    features = get_valid_features()
    features.pop(feature_names[0])

    response = client.post(
        "/predict",
        json={"features": features}
    )

    assert response.status_code == 422


def test_invalid_feature_type():
    features = get_valid_features()
    features[feature_names[0]] = "invalid"

    response = client.post(
        "/predict",
        json={"features": features}
    )

    assert response.status_code == 422
def test_negative_feature_value():
    features = get_valid_features()
    features[feature_names[0]] = -1

    response = client.post(
        "/predict",
        json={"features": features}
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Feature values cannot be negative"