
import argparse
import time
import os
from pathlib import Path

import yaml
import joblib
import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from src.data import load_data, split_data
from src.evaluate import evaluate_model


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main(config_path):

    # ---------------------------------------------------------
    # 1. Load configuration
    # ---------------------------------------------------------
    config = load_config(config_path)

    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]

    # ---------------------------------------------------------
    # 2. Configure MLflow tracking
    # ---------------------------------------------------------
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    print("MLflow Tracking URI:", mlflow.get_tracking_uri())

    mlflow.set_experiment("MLOps-Lab-3-4")

    # ---------------------------------------------------------
    # 3. Create required directories
    # ---------------------------------------------------------
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 4. Load data
    # ---------------------------------------------------------
    X, y = load_data(data_config["path"])

    print("Dataset loaded successfully.")
    print("Number of samples:", len(X))
    print("Number of features:", X.shape[1])

    # ---------------------------------------------------------
    # 5. Split dataset
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        training_config["test_size"],
        training_config["random_state"]
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # ---------------------------------------------------------
    # 6. Start MLflow run
    # ---------------------------------------------------------
    with mlflow.start_run() as run:

        start_time = time.time()

        # -----------------------------------------------------
        # 7. Create model
        # -----------------------------------------------------
        model = RandomForestClassifier(
            n_estimators=model_config["n_estimators"],
            max_depth=model_config["max_depth"],
            min_samples_split=model_config["min_samples_split"],
            random_state=training_config["random_state"]
        )

        # -----------------------------------------------------
        # 8. Train model
        # -----------------------------------------------------
        print("\nTraining model...")
        model.fit(X_train, y_train)

        training_time = time.time() - start_time

        # -----------------------------------------------------
        # 9. Evaluate model
        # -----------------------------------------------------
        metrics, cm, report = evaluate_model(
            model,
            X_test,
            y_test
        )

        print("\nEvaluation Results")
        print("------------------")
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1-score : {metrics['f1_score']:.4f}")
        print(f"Training Time: {training_time:.4f} seconds")

        # -----------------------------------------------------
        # 10. Save confusion matrix
        # -----------------------------------------------------
        confusion_matrix_path = (
            "reports/figures/confusion_matrix.png"
        )

        ConfusionMatrixDisplay(
            confusion_matrix=cm
        ).plot()

        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(confusion_matrix_path)
        plt.close()

        # -----------------------------------------------------
        # 11. Save classification report
        # -----------------------------------------------------
        classification_report_path = (
            "reports/classification_report.txt"
        )

        with open(classification_report_path, "w") as file:
            file.write(report)

        # -----------------------------------------------------
        # 12. Save evaluation results
        # -----------------------------------------------------
        results_path = "reports/evaluation_results.csv"

        with open(results_path, "w") as file:
            file.write("Metric,Value\n")
            file.write(f"Accuracy,{metrics['accuracy']}\n")
            file.write(f"Precision,{metrics['precision']}\n")
            file.write(f"Recall,{metrics['recall']}\n")
            file.write(f"F1-score,{metrics['f1_score']}\n")
            file.write(f"Training_Time,{training_time}\n")

        # -----------------------------------------------------
        # 13. Save model
        # -----------------------------------------------------
        model_path = "models/model.joblib"
        joblib.dump(model, model_path)

        # -----------------------------------------------------
        # 14. Log MLflow parameters
        # -----------------------------------------------------
        mlflow.log_param("model_type", model_config["type"])
        mlflow.log_param(
            "n_estimators",
            model_config["n_estimators"]
        )
        mlflow.log_param(
            "max_depth",
            model_config["max_depth"]
        )
        mlflow.log_param(
            "test_size",
            training_config["test_size"]
        )
        mlflow.log_param(
            "random_state",
            training_config["random_state"]
        )
        mlflow.log_param(
            "min_samples_split",
            model_config["min_samples_split"]
        )

        # -----------------------------------------------------
        # 15. Log MLflow metrics
        # -----------------------------------------------------
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1_score", metrics["f1_score"])
        mlflow.log_metric("training_time", training_time)

        # -----------------------------------------------------
        # 16. Log artifacts
        # -----------------------------------------------------
        mlflow.log_artifact(confusion_matrix_path)
        mlflow.log_artifact(classification_report_path)
        mlflow.log_artifact(results_path)

        # -----------------------------------------------------
        # 17. Log trained model
        # -----------------------------------------------------
        mlflow.sklearn.log_model(
            model,
            name="random_forest_model"
        )

        print("\nMLflow Run ID:", run.info.run_id)

        print("\nTraining completed successfully.")
        print("Model saved to:", model_path)
        print("Confusion matrix saved to:", confusion_matrix_path)
        print("Classification report saved to:", classification_report_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Reproducible ML training with MLflow"
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    main(args.config)
