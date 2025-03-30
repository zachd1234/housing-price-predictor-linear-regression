import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import numpy as np
import scipy.stats as stats

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame  

# Define independent (X) and dependent (Y) variables
X = df[["MedInc"]].values  # Convert to NumPy array
Y = df["MedHouseVal"].values  # Convert to NumPy array

# Apply log transformation to Y
Y_log = np.log(Y)

# Train a Linear Regression Model on log-transformed Y
model = LinearRegression()
model.fit(X, Y_log)

# Generate predictions for regression line (in log scale)
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
Y_pred_log = model.predict(X_range)

# Convert predictions back to original scale
Y_pred = np.exp(Y_pred_log)

# Compute Confidence Intervals
n = len(X)  # Number of data points
X_mean = np.mean(X)  # Mean of X
t_value = stats.t.ppf(0.975, df=n-2)  # 95% Confidence level (two-tailed)
s_err = np.sqrt(np.sum((Y_log - model.predict(X))**2) / (n - 2))  # Standard error in log scale

# Compute margin of error in log scale
margin_error_log = t_value * s_err * np.sqrt(1/n + (X_range - X_mean)**2 / np.sum((X - X_mean)**2))

# Convert confidence intervals back to original scale
upper_bound = np.exp(Y_pred_log + margin_error_log.flatten())
lower_bound = np.exp(Y_pred_log - margin_error_log.flatten())

# Scatter plot of original data
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["MedInc"], y=df["MedHouseVal"], alpha=0.5, label="Data")

# Regression Line
plt.plot(X_range, Y_pred, color="red", linewidth=2, label="Regression Line (Log-Transformed)")

# Confidence Interval Shading
plt.fill_between(X_range.flatten(), lower_bound, upper_bound, color="red", alpha=0.2, label="95% Confidence Interval")

# Titles and labels
plt.title("Log-Transformed Regression: Income vs. House Value", fontsize=14)
plt.xlabel("Median Income (in $10,000s)", fontsize=12)
plt.ylabel("Median House Value (in $100,000s)", fontsize=12)
plt.legend()

# Show the plot
plt.show()
