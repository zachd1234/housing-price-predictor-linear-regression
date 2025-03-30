import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import numpy as np

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame  

# Define independent (X) and dependent (Y) variables
X = df[["MedInc"]]  # Independent variable (reshaped for sklearn)
Y = df["MedHouseVal"]  # Dependent variable

# Train a Linear Regression Model
model = LinearRegression()
model.fit(X, Y)

# Generate predictions for regression line
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)  # Creates X values for line
Y_pred = model.predict(X_range)  # Predict Y values for regression line

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["MedInc"], y=df["MedHouseVal"], alpha=0.5, label="Data")

# Regression Line
plt.plot(X_range, Y_pred, color="red", linewidth=2, label="Regression Line")

# Titles and labels
plt.title("Scatter Plot with Regression Line: Income vs. House Value", fontsize=14)
plt.xlabel("Median Income (in $10,000s)", fontsize=12)
plt.ylabel("Median House Value (in $100,000s)", fontsize=12)
plt.legend()

# Show the plot
plt.show()
