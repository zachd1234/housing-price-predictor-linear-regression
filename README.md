# California Housing Price Predictor 🏡

This project predicts median house prices in California using a linear regression model, with extensive data preprocessing, statistical analysis, and model evaluation.

It includes:
- Outlier handling (Winsorization)
- Log-transformations for better linearity
- Full regression training and evaluation
- Confidence interval visualization
- A terminal UI that lets users predict housing prices based on real features

---

## 🧠 Project Highlights

- 📈 **Regression Model**: Trained on California Housing dataset using a `StandardScaler` + `LinearRegression` pipeline.
- 📊 **Data Analysis**:
  - Histograms of original vs. log-transformed variables
  - Outlier clipping to improve model robustness
  - Confidence intervals around regression predictions
- 🖥️ **Terminal UI**:
  - Users input real-world data (income, house age, population, etc.)
  - The model predicts estimated median house values instantly
- 🧮 **Feature Importance**:
  - Coefficients extracted to interpret the most influential features
- 📈 **Evaluation Metrics**:
  - RMSE
  - R² Score (coefficient of determination)

---

## 🛠️ Tech Stack

- Python 3
- NumPy
- Pandas
- Scikit-learn
- Seaborn & Matplotlib
- SciPy

🏠 California Housing Price Predictor 🏠
Please enter the following information:
Median Income in block group (tens of thousands): 5
Median House Age in block group (years): 25
...

🔮 Prediction Result:
Estimated Median House Value: $356,478.35
