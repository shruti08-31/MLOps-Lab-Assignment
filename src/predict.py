import joblib
import pandas as pd


def predict(model_path, data_path):

    model = joblib.load(model_path)

    data = pd.read_csv(data_path)

    X = data.drop("target", axis=1)

    predictions = model.predict(X)

    print("Predictions generated successfully.")
    print("Number of predictions:", len(predictions))

    print("\nFirst 10 predictions:")
    print(predictions[:10])

    return predictions


if __name__ == "__main__":

    predict(
        "models/model.joblib",
        "data/raw/breast_cancer.csv"
    )