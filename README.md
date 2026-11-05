\# MLOps Lab 3 - Reproducible Machine Learning \& MLflow



\## Objective



The objective of this project is to create a reproducible machine learning workflow using configuration management, isolated environments, random seeds, MLflow experiment tracking, and Git.



\## Dataset



Breast Cancer Wisconsin dataset provided by Scikit-learn.



\## Model



Random Forest Classifier.



\## Reproducibility



The project controls randomness using a fixed random\_state and stores model and training parameters in config.yaml.



\## MLflow Tracking



Each training execution records:



\- Model type

\- Number of estimators

\- Maximum depth

\- Test size

\- Random state

\- Accuracy

\- Precision

\- Recall

\- F1-score

\- Confusion matrix

\- Classification report

\- Trained model



\## How to Run



Activate the virtual environment:



&#x20;   .venv\\Scripts\\activate



Install dependencies:



&#x20;   pip install -r requirements.txt



Run training:



&#x20;   python src/train.py --config configs/config.yaml



Start MLflow:



&#x20;   mlflow ui



Open the MLflow interface in a browser.



\## Project Structure



```text

mlops-lab-3-4/

├── data/

├── models/

├── notebooks/

├── src/

├── configs/

├── reports/

├── screenshots/

├── requirements.txt

└── README.md

