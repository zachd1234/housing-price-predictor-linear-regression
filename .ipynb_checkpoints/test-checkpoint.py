from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset
california = fetch_california_housing(as_frame=True)

# Convert to DataFrame
df = california.frame

# Display first 5 rows
print(df.head())
