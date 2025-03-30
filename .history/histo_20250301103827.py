import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame  

# Create Histogram
plt.figure(figsize=(8, 6))
sns.histplot(df["MedHouseVal"], bins=30, kde=True)  # KDE=True adds a smooth density curve

# Titles and labels
plt.xlabel("Median House Value (in $100,000s)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Distribution of House Prices", fontsize=14)

# Show the plot
plt.show()
