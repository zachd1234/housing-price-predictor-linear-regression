import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Load dataset
california = fetch_california_housing(as_frame=True)

# Convert to DataFrame
df = california.frame

# Compute correlation matrix
corr_matrix = df.corr()

# Create heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("California Housing Data Correlation Heatmap")
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x=df["MedInc"], y=df["MedHouseVal"])
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Income vs House Value")
plt.show()
