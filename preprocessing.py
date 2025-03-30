# Import libraries
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 🔹 Step 1: Load Dataset
california = fetch_california_housing(as_frame=True)
df = california.frame  # Convert to DataFrame

# 🔹 Step 2: Handle Outliers (Winsorization)
for feature in ["MedHouseVal", "MedInc"]:
    df[feature] = np.clip(df[feature], df[feature].quantile(0.01), df[feature].quantile(0.99))

# 🔹 Step 3: Log Transformation (Only on Target Variable)
df["MedHouseVal"] = np.log1p(df["MedHouseVal"])  # Log transformation for better linearity

# 🔹 Step 4: Split into Features (X) and Target (y)
X = df.drop("MedHouseVal", axis=1)  # All features except target
y = df["MedHouseVal"]  # Target variable (log-transformed house price)

# 🔹 Step 5: Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Step 6: Create Pipeline (Standardization + Regression Model)
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardizes all features
    ('regressor', LinearRegression())  # Trains the regression model
])

# 🔹 Step 7: Train the Model
pipeline.fit(X_train, y_train)

# 🔹 Step 8: Evaluate Model Performance
# Make predictions on test set
y_pred = pipeline.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5  # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation:")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# 🔹 Step 9: Get Feature Importance (Coefficients)
coefficients = pipeline.named_steps['regressor'].coef_
feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})
feature_importance = feature_importance.sort_values(by="Coefficient", ascending=False)

# 🔹 Step 10: Print Model Summary
print("🔹 Training Complete!")
print("📊 Feature Importance (Which Features Affect House Prices the Most?)")
print(feature_importance)

# 🔹 Step 11: Create Terminal UI for Predictions
def get_user_input():
    """Get housing information from user via terminal input"""
    print("\n🏠 California Housing Price Predictor 🏠")
    print("Please enter the following information:")
    
    features = {}
    features["MedInc"] = float(input("Median Income in block group (tens of thousands): "))
    features["HouseAge"] = float(input("Median House Age in block group (years): "))
    features["AveRooms"] = float(input("Average Rooms per household: "))
    features["AveBedrms"] = float(input("Average Bedrooms per household: "))
    features["Population"] = float(input("Block group population: "))
    features["AveOccup"] = float(input("Average Occupants per household: "))
    features["Latitude"] = float(input("Block group latitude: "))
    features["Longitude"] = float(input("Block group longitude: "))
    
    return pd.DataFrame([features])

def predict_house_price(input_data):
    """Predict house price based on input data"""
    # Make prediction using our pipeline
    log_prediction = pipeline.predict(input_data)[0]
    
    # Convert from log-transformed value back to original scale
    prediction = np.expm1(log_prediction)
    
    return prediction

def run_prediction_ui():
    """Run the terminal UI for predictions"""
    while True:
        try:
            # Get user input
            user_data = get_user_input()
            
            # Make prediction
            predicted_price = predict_house_price(user_data)
            
            # Display result
            print("\n🔮 Prediction Result:")
            print(f"Estimated Median House Value: ${predicted_price:.2f}k")
            
            # Ask if user wants to make another prediction
            again = input("\nWould you like to make another prediction? (y/n): ")
            if again.lower() != 'y':
                print("Thank you for using the California Housing Price Predictor!")
                break
                
        except ValueError:
            print("Error: Please enter valid numeric values.")
        except Exception as e:
            print(f"An error occurred: {e}")

# Run the prediction UI if this script is executed directly
if __name__ == "__main__":
    run_prediction_ui()
