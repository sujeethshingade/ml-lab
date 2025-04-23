import warnings
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load Boston Housing dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
boston_df = pd.read_csv(url)

# Print column names
print("Available columns in the dataset:")
print(boston_df.columns.tolist())

warnings.filterwarnings('ignore')

# Part 1: Linear Regression with Boston Housing Dataset
print("Part 1: Linear Regression - Boston Housing Dataset")
print("-" * 50)

# Load Boston Housing dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
boston_df = pd.read_csv(url)

# Features and target (using correct column names)
X_boston = boston_df.drop('medv', axis=1)  # All columns except target
y_boston = boston_df['medv']  # median house value

# Print dataset info
print("\nDataset Information:")
print(f"Number of samples: {len(X_boston)}")
print(f"Number of features: {len(X_boston.columns)}")
print("\nFeatures:")
for name in X_boston.columns:
    print(f"- {name}")

# Split the data
X_train_boston, X_test_boston, y_train_boston, y_test_boston = train_test_split(
    X_boston, y_boston, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_boston_scaled = scaler.fit_transform(X_train_boston)
X_test_boston_scaled = scaler.transform(X_test_boston)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_boston_scaled, y_train_boston)

# Make predictions
y_pred_boston = lr_model.predict(X_test_boston_scaled)

# Calculate metrics
mse_boston = mean_squared_error(y_test_boston, y_pred_boston)
rmse_boston = np.sqrt(mse_boston)
r2_boston = r2_score(y_test_boston, y_pred_boston)

print("\nLinear Regression Results:")
print(f"Mean Squared Error: {mse_boston:.2f}")
print(f"Root Mean Squared Error: {rmse_boston:.2f}")
print(f"R² Score: {r2_boston:.2f}")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': X_boston.columns,
    'Coefficient': lr_model.coef_
})
feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values(
    'Abs_Coefficient', ascending=False)

print("\nFeature Importance:")
print(feature_importance[['Feature', 'Coefficient']].to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(12, 6))
plt.bar(feature_importance['Feature'], feature_importance['Coefficient'])
plt.xticks(rotation=45)
plt.title('Feature Importance in Boston Housing Price Prediction')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.tight_layout()
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_boston, y_pred_boston, alpha=0.5)
plt.plot([y_test_boston.min(), y_test_boston.max()], [
         y_test_boston.min(), y_test_boston.max()], 'r--', lw=2)
plt.xlabel('Actual Prices ($1000s)')
plt.ylabel('Predicted Prices ($1000s)')
plt.title('Actual vs Predicted Housing Prices')
plt.tight_layout()
plt.show()

# Part 2: Polynomial Regression with Auto MPG Dataset
print("\nPart 2: Polynomial Regression - Auto MPG Dataset")
print("-" * 50)

# Load Auto MPG dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin', 'Car Name']
df = pd.read_csv(url, names=column_names, delim_whitespace=True)

# Clean the data
df = df.replace('?', np.nan)
df = df.dropna()
df['Horsepower'] = df['Horsepower'].astype(float)

# Select features for polynomial regression
X_mpg = df[['Horsepower']].values
y_mpg = df['MPG'].values

# Scale features for polynomial regression
scaler_mpg = StandardScaler()
X_mpg_scaled = scaler_mpg.fit_transform(X_mpg)

# Split the data
X_train_mpg, X_test_mpg, y_train_mpg, y_test_mpg = train_test_split(
    X_mpg_scaled, y_mpg, test_size=0.2, random_state=42
)

# Create and train models with different polynomial degrees
degrees = [1, 2, 3]
plt.figure(figsize=(15, 5))

for i, degree in enumerate(degrees, 1):
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train_mpg)
    X_test_poly = poly_features.transform(X_test_mpg)

    # Train model
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train_mpg)

    # Make predictions
    y_pred_poly = poly_model.predict(X_test_poly)

    # Calculate metrics
    mse_poly = mean_squared_error(y_test_mpg, y_pred_poly)
    rmse_poly = np.sqrt(mse_poly)
    r2_poly = r2_score(y_test_mpg, y_pred_poly)

    print(f"\nPolynomial Regression (degree {degree}) Results:")
    print(f"Mean Squared Error: {mse_poly:.2f}")
    print(f"Root Mean Squared Error: {rmse_poly:.2f}")
    print(f"R² Score: {r2_poly:.2f}")

    # Plot results
    plt.subplot(1, 3, i)
    plt.scatter(X_test_mpg, y_test_mpg, color='blue',
                alpha=0.5, label='Actual')

    # Sort points for smooth curve
    X_sort = np.sort(X_test_mpg, axis=0)
    X_sort_poly = poly_features.transform(X_sort)
    y_sort_pred = poly_model.predict(X_sort_poly)

    plt.plot(X_sort, y_sort_pred, color='red', label='Predicted')
    plt.xlabel('Horsepower (scaled)')
    plt.ylabel('MPG')
    plt.title(f'Polynomial Regression (degree {degree})')
    plt.legend()

plt.tight_layout()
plt.show()
