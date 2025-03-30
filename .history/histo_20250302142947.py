import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import fetch_california_housing

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame  

# Apply Log Transformation
df["LogMedHouseVal"] = np.log(df["MedHouseVal"])
df["LogMedInc"] = np.log(df["MedInc"])

# Calculate medians after log transformation
median_log_house_val = df["LogMedHouseVal"].median()
median_log_income = df["LogMedInc"].median()

# Calculate medians for original values
median_house_val = df["MedHouseVal"].median()
median_income = df["MedInc"].median()

# Create Subplots for Original vs. Log-Transformed Histograms
fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 2 rows, 2 columns

# 🔹 Original Histogram of House Prices
sns.histplot(df["MedHouseVal"], bins=30, kde=True, ax=axes[0, 0], color='blue', alpha=0.6)
axes[0, 0].axvline(df["MedHouseVal"].median(), color='red', linestyle='dashed', linewidth=2, label=f'Median: {median_house_val:.2f}')
axes[0, 0].set_xlabel("Median House Value (in $100,000s)", fontsize=12)
axes[0, 0].set_ylabel("Frequency", fontsize=12)
axes[0, 0].set_title("Original Distribution: House Prices", fontsize=14)
axes[0, 0].legend()

# 🔹 Log-Transformed Histogram of House Prices
sns.histplot(df["LogMedHouseVal"], bins=30, kde=True, ax=axes[0, 1], color='blue', alpha=0.6)
axes[0, 1].axvline(median_log_house_val, color='red', linestyle='dashed', linewidth=2, label=f'Median: {median_log_house_val:.2f}')
axes[0, 1].set_xlabel("Log of Median House Value", fontsize=12)
axes[0, 1].set_ylabel("Frequency", fontsize=12)
axes[0, 1].set_title("Log-Transformed: House Prices", fontsize=14)
axes[0, 1].legend()

# 🔹 Original Histogram of Median Income
sns.histplot(df["MedInc"], bins=30, kde=True, ax=axes[1, 0], color='green', alpha=0.6)
axes[1, 0].axvline(df["MedInc"].median(), color='red', linestyle='dashed', linewidth=2, label=f'Median: {median_income:.2f}')
axes[1, 0].set_xlabel("Median Income (in $10,000s)", fontsize=12)
axes[1, 0].set_ylabel("Frequency", fontsize=12)
axes[1, 0].set_title("Original Distribution: Median Income", fontsize=14)
axes[1, 0].legend()

# 🔹 Log-Transformed Histogram of Median Income
sns.histplot(df["LogMedInc"], bins=30, kde=True, ax=axes[1, 1], color='green', alpha=0.6)
axes[1, 1].axvline(median_log_income, color='red', linestyle='dashed', linewidth=2, label=f'Median: {median_log_income:.2f}')
axes[1, 1].set_xlabel("Log of Median Income", fontsize=12)
axes[1, 1].set_ylabel("Frequency", fontsize=12)
axes[1, 1].set_title("Log-Transformed: Median Income", fontsize=14)
axes[1, 1].legend()

# Show the plot
plt.tight_layout()
plt.show()
