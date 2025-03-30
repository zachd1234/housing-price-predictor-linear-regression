import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame  

# Scatter plot: Income vs. House Value
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["MedInc"], y=df["MedHouseVal"], alpha=0.5)

# Titles and labels
plt.title("Scatter Plot: Income vs. House Value", fontsize=14)
plt.xlabel("Median Income (in $10,000s)", fontsize=12)
plt.ylabel("Median House Value (in $100,000s)", fontsize=12)

# Show the plot
plt.show()
