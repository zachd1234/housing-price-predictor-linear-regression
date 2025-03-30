from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
california = fetch_california_housing(as_frame=True)

# Convert to DataFrame
df = california.frame

# Handle outliers by capping extreme values (Winsorization method)
df["MedHouseVal"] = np.clip(df["MedHouseVal"], df["MedHouseVal"].quantile(0.01), df["MedHouseVal"].quantile(0.99))
df["MedInc"] = np.clip(df["MedInc"], df["MedInc"].quantile(0.01), df["MedInc"].quantile(0.99))

# Apply log transformation to target variable (House Prices)
df["MedHouseVal"] = np.log1p(df["MedHouseVal"])  # log1p avoids issues with zeros

# Split dataset into features (X) and target variable (y)
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Normalize numerical features (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Print dataset information
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
print(f"Features after scaling: {X_train.shape[1]}")
