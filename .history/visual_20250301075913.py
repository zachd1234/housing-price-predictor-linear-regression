import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load your data into a DataFrame
df = pd.read_csv('your_data_file.csv')  # Replace with your actual data source

# Compute correlation matrix
corr_matrix = df.corr()

# Create heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
