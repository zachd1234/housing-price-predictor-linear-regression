# Import libraries
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# 🔹 Step 1: Load Dataset
california = fetch_california_housing(as_frame=True)
df = california.frame  # Convert to DataFrame

# 🔹 Step 2: Handle Outliers (Winsorization)
for feature in ["MedHouseVal", "MedInc"]:
    df[feature] = np.clip(df[feature], df[feature].quantile(0.01), df[feature].quantile(0.99))

# 🔹 Step 3: Log Transformation (Only on Target Variable)
df["MedHouseVal"] = np.log1p(df["MedHouseVal"])  # Log transformation for better linearity

# 🔹 Step 4: Split into Features (X) and Target (y)
X = df.drop("MedHouseVal", axis=1)  # All features except target
y = df["MedHouseVal"]  # Target variable (log-transformed house price)

# 🔹 Step 5: Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Step 6: Create Pipeline (Standardization + Regression Model)
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardizes all features
    ('regressor', LinearRegression())  # Trains the regression model
])

# 🔹 Step 7: Train the Model
pipeline.fit(X_train, y_train)

# 🔹 Step 8: Get Feature Importance (Coefficients)
coefficients = pipeline.named_steps['regressor'].coef_
feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})
feature_importance = feature_importance.sort_values(by="Coefficient", ascending=False)

# 🔹 Step 9: Print Model Summary
print("🔹 Training Complete!")
print("📊 Feature Importance (Which Features Affect House Prices the Most?)")
print(feature_importance)
