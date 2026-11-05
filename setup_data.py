from sklearn.datasets import load_breast_cancer
import pandas as pd
from pathlib import Path

# Load the Breast Cancer Wisconsin dataset
dataset = load_breast_cancer(as_frame=True)

# Convert dataset to DataFrame
df = dataset.frame

# Create the raw data directory
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

# Save dataset
output_file = output_dir / "breast_cancer.csv"
df.to_csv(output_file, index=False)

print("Dataset created successfully.")
print("Dataset shape:", df.shape)
print("Saved to:", output_file)

print("\nFirst 5 rows:")
print(df.head())