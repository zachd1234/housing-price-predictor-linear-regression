import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame  

# Calculate median values
median_house_val = df["MedHouseVal"].median()
median_income = df["MedInc"].median()

# Create Histogram
plt.figure(figsize=(8, 6))
sns.histplot(df["MedHouseVal"], bins=30, kde=True, alpha=0.6)  # KDE=True adds a smooth density curve

# Add vertical line for median house value
plt.axvline(median_house_val, color='red', linestyle='dashed', linewidth=2, label=f'Median House Value: {median_house_val:.2f}')

# Add text annotation for median house value
plt.text(median_house_val, plt.ylim()[1] * 0.9, f'Median House Value\n${median_house_val * 100000:.0f}', 
         color='red', ha='right', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

# Add annotation for median income (not plotted, but displayed)
plt.figtext(0.15, 0.8, f'Median Income: ${median_income * 10000:.0f}', fontsize=12, color='blue', bbox=dict(facecolor='white', alpha=0.5))

# Titles and labels
plt.xlabel("Median House Value (in $100,000s)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Distribution of House Prices", fontsize=14)
plt.legend()

# Show the plot
plt.show()
