import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame  

# Calculate medians
median_house_val = df["MedHouseVal"].median()
median_income = df["MedInc"].median()

# Create Subplots for Two Histograms
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # Two side-by-side histograms

# Histogram of House Prices
sns.histplot(df["MedHouseVal"], bins=30, kde=True, ax=axes[0], color='blue', alpha=0.6)
axes[0].axvline(median_house_val, color='red', linestyle='dashed', linewidth=2, label=f'Median: {median_house_val:.2f}')
axes[0].set_xlabel("Median House Value (in $100,000s)", fontsize=12)
axes[0].set_ylabel("Frequency", fontsize=12)
axes[0].set_title("Distribution of House Prices", fontsize=14)
axes[0].legend()

# Histogram of Median Income
sns.histplot(df["MedInc"], bins=30, kde=True, ax=axes[1], color='green', alpha=0.6)
axes[1].axvline(median_income, color='red', linestyle='dashed', linewidth=2, label=f'Median: {median_income:.2f}')
axes[1].set_xlabel("Median Income (in $10,000s)", fontsize=12)
axes[1].set_ylabel("Frequency", fontsize=12)
axes[1].set_title("Distribution of Median Income", fontsize=14)
axes[1].legend()

# Show the plot
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()
