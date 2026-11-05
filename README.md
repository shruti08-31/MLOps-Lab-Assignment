# MLOps Lab Assignment – ML Pipelines, FastAPI and Docker

## 1. Project Overview

This project demonstrates an end-to-end Machine Learning Operations (MLOps) workflow using the Breast Cancer Wisconsin dataset.

The project focuses on building a reproducible machine learning pipeline and deploying the trained model through a REST API and Docker containers.

The implementation covers:

- Data loading, validation and splitting
- Modular machine learning pipeline
- Configuration-based model training
- Reproducibility using fixed random seeds
- MLflow experiment tracking
- Model evaluation and prediction
- FastAPI model serving
- Automated testing using Pytest
- Docker image creation and containerization
- Docker Compose for multi-service execution
- DVC for dataset versioning
- Git for source code version control

## 2. Dataset

**Dataset:** Breast Cancer Wisconsin Dataset  
**Source:** Scikit-learn

Dataset information:

| Property | Value |
|---|---:|
| Total records | 569 |
| Input features | 30 |
| Target column | target |
| Class 0 | 212 |
| Class 1 | 357 |

The dataset contains numerical features and does not contain missing values.

The dataset is divided into:
- 80% training data
- 20% testing data

Stratified splitting is used to maintain the class distribution in both sets.

## 3. Machine Learning Model

**Algorithm:** Random Forest Classifier

The model parameters are maintained in `configs/config.yaml`.

| Parameter | Value |
|---|---:|
| Number of estimators | 200 |
| Maximum depth | None |
| Minimum samples split | 2 |
| Test size | 0.2 |
| Random state | 42 |

Random Forest was selected for the classification task. Since the dataset contains numerical features without missing values, and Random Forest does not require feature scaling, no additional imputation, encoding or scaling is required.

## 4. Project Structure

```text
MLOps-Lab-Assignment/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── model.joblib
│
├── reports/
│   ├── classification_report.txt
│   ├── evaluation_results.csv
│   └── figures/
│       └── confusion_matrix.png
│
├── configs/
│   └── config.yaml
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── api.py
│
├── tests/
│   ├── test_api.py
│   └── test_data.py
│
├── screenshots/
│
├── .dvc/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── Dockerfile.mlflow
├── docker-compose.yml
├── requirements.txt
├── requirements-docker.txt
├── sample_request.json
└── README.md
```

## 5. Environment Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```cmd
.venv\Scripts\activate
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## 6. Data Processing and Model Training

The modular pipeline separates data loading, training, evaluation and prediction into individual Python modules.

### Run model training

From the project root directory:

```bash
python -m src.train --config configs/config.yaml
```

The training process performs the following operations:

1. Loads the dataset.
2. Validates the input data.
3. Separates input features and target.
4. Splits the dataset into training and testing sets.
5. Trains the Random Forest classifier.
6. Evaluates the model on held-out test data.
7. Saves the trained model.
8. Saves evaluation reports and confusion matrix.
9. Logs the experiment to MLflow when tracking is configured.

### Generated files

- `models/model.joblib`
- `reports/classification_report.txt`
- `reports/evaluation_results.csv`
- `reports/figures/confusion_matrix.png`

## 7. Model Evaluation

The model was evaluated using accuracy, precision, recall and F1-score.

### Evaluation Results

| Metric | Result |
|---|---:|
| Accuracy | 95.61% |
| Precision | 95.89% |
| Recall | 97.22% |
| F1-score | 96.55% |
| Training time | 0.2273 seconds |

These results were obtained from the held-out test dataset.

## 8. MLflow Experiment Tracking

MLflow is used to track model training experiments and maintain experiment information.

The following information is logged:

**Parameters**
- Model type
- Number of estimators
- Maximum depth
- Test size
- Random state

**Metrics**
- Accuracy
- Precision
- Recall
- F1-score
- Training time

**Artifacts**
- Confusion matrix
- Classification report
- Evaluation results
- Trained model

### Start MLflow

The project includes an MLflow service through Docker Compose.

Start the service:

```cmd
docker compose up -d mlflow
```

Configure the tracking URI:

```cmd
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Run training:

```cmd
python -m src.train --config configs/config.yaml
```

Open the MLflow interface:

http://127.0.0.1:5000

## 9. Model Prediction

The prediction module loads the saved model and performs predictions.

Run:

```bash
python -m src.predict
```

The model can also be loaded independently using Joblib for inference in another Python process.

## 10. FastAPI Model Serving

FastAPI is used to expose the trained machine learning model through REST API endpoints.

### Start the API

```cmd
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

### API documentation

Open the following URL in a browser:

http://127.0.0.1:8000/docs

### Available endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks API health |
| POST | `/predict` | Generates a prediction |

### Health check response

```json
{
  "status": "healthy",
  "model": "RandomForestClassifier"
}
```

### Prediction request

The `/predict` endpoint accepts a JSON request containing the 30 input features.

The project includes `sample_request.json`, which can be used to test the endpoint through Swagger UI or curl.

Example request structure:

```json
{
  "features": {
    "feature_name": 10.5
  }
}
```

The actual request must contain all 30 expected feature names with numerical values. Use `sample_request.json` for a complete valid request.

### Example prediction response

```json
{
  "prediction": 0,
  "probability": 0.95
}
```

The prediction and probability depend on the submitted input.

The API validates missing features, extra features, invalid numerical values and negative values.

## 11. Automated Testing

Pytest is used to test the data processing modules and API functionality.

Run the tests:

```cmd
python -m pytest tests -v
```

The test suite contains seven tests covering:

**Data tests**
- Data loading
- Data splitting

**API tests**
- Health endpoint
- Valid prediction
- Missing features
- Invalid feature type
- Negative feature values

**Verified result: 7 passed.**

## 12. Docker Containerization

Docker is used to package the FastAPI application and its required dependencies into a container.

### Build Docker image

```cmd
docker build -t ml-inference-api:1.0 .
```

### Run Docker container

```cmd
docker run --rm -p 8000:8000 ml-inference-api:1.0
```

### Test the health endpoint

```cmd
curl http://127.0.0.1:8000/health
```

### Test prediction

```cmd
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```

The Docker image uses `requirements-docker.txt` to install the runtime dependencies required by the API.

## 13. Docker Compose

Docker Compose is used to run the API and MLflow services together.

The project contains two services:

1. **API service:** Provides machine learning predictions through FastAPI.
2. **MLflow service:** Provides experiment tracking and stores experiment data.

### Start services

```cmd
docker compose up -d --build
```

### Check running containers

```cmd
docker compose ps
```

### View logs

```cmd
docker compose logs
```

### Stop services

```cmd
docker compose down
```

The MLflow service uses a named Docker volume to preserve its tracking data when containers are stopped and recreated.

## 14. DVC Dataset Versioning

DVC (Data Version Control) is used to track the dataset separately from Git.

### Initialize DVC

```cmd
dvc init
```

### Track the dataset

```cmd
dvc add data/raw/breast_cancer.csv
```

### Check DVC status

```cmd
dvc status
```

### Restore tracked data

```cmd
dvc checkout
```

DVC stores the dataset in its local cache and creates a `.dvc` metadata file. Git tracks the metadata instead of directly tracking the dataset.

**Note:** A remote DVC storage has not been configured. Dataset restoration was tested successfully using the local DVC cache.

## 15. Local and Docker Prediction Comparison

The same sample request was submitted to the locally running FastAPI application and the Dockerized API.

The responses were saved and compared using:

```cmd
fc docker_response.json local_response.json
```

The comparison returned:

```text
FC: no differences encountered
```

This verified that both environments generated identical responses for the tested input.

## 16. Online and Batch Inference

**Online inference**

The FastAPI application accepts an individual prediction request and returns the result immediately through the `/predict` endpoint.

**Batch inference**

The prediction module loads the saved model and performs predictions on multiple dataset records.

Both approaches use the same trained Random Forest model.

## 17. Reproducibility

The project supports reproducible machine learning through:

- Centralized configuration using YAML
- Fixed random state
- Stratified train-test splitting
- Version-controlled source code using Git
- Dataset versioning using DVC
- MLflow experiment tracking
- Saved model artifacts
- Dependency files
- Docker-based execution

## 18. Conclusion

This project demonstrates the development and deployment of a reproducible machine learning workflow.

It integrates data processing, model training, evaluation, experiment tracking, API development, automated testing, Docker containerization and dataset versioning.

The comparison between local and Docker predictions also verifies consistent inference results for the tested sample.
