from pathlib import Path
import math
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

model = joblib.load(MODEL_PATH)
feature_names = list(model.feature_names_in_)

app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API for Random Forest breast cancer classification",
    version="1.0"
)


class PredictionRequest(BaseModel):
    features: dict[str, float]


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "RandomForestClassifier"
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    features = request.features

    missing = [x for x in feature_names if x not in features]
    extra = [x for x in features if x not in feature_names]

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Incorrect feature names",
                "missing_features": missing,
                "extra_features": extra
            }
        )

    values = list(features.values())

    if not all(math.isfinite(x) for x in values):
        raise HTTPException(
            status_code=422,
            detail="Feature values must be finite numbers"
        )

    if any(x < 0 for x in values):
        raise HTTPException(
            status_code=422,
            detail="Feature values cannot be negative"
        )

    data = pd.DataFrame(
        [[features[x] for x in feature_names]],
        columns=feature_names
    )

    prediction = int(model.predict(data)[0])
    probability = float(
        model.predict_proba(data)[0][prediction]
    )

    return {
        "prediction": prediction,
        "probability": round(probability, 4)
    }