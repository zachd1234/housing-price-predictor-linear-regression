from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset
california = fetch_california_housing(as_frame=True)

# Convert to DataFrame
df = california.frame

# Display first 5 rows
print(df.head())

# Check dataset info
print(df.info())

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())
