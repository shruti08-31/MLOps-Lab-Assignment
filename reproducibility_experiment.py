from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Running the same training process 3 times WITHOUT fixing a random seed.")
print()

for run in range(1, 4):

    # Random train-test split
    # No random_state is used
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2
    )

    # Random Forest
    # No random_state is used
    model = RandomForestClassifier(
        n_estimators=100
    )

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Display results
    print(f"Run {run}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("-" * 40)