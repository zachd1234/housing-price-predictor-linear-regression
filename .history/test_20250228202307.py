from sklearn.datasets import load_boston
import pandas as pd

# Load dataset
boston = load_boston()

# Convert to DataFrame
df = pd.DataFrame(boston.data, columns=boston.feature_names)

# Add target variable (house prices)
df['PRICE'] = boston.target

# Display first 5 rows
df.head()
