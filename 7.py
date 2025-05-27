import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def model(file, x, y, title, poly=False):
    df = pd.read_csv(file).dropna()
    X, Y = df[[x]], df[y]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

    reg = make_pipeline(PolynomialFeatures(2), StandardScaler(), LinearRegression()) if poly else LinearRegression()
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)

    plt.scatter(X_test, Y_test, label="Actual", alpha=0.5)
    plt.scatter(X_test, Y_pred, label="Predicted", alpha=0.5)
    plt.title(title); plt.legend(); plt.grid(); plt.show()

    print(f"{title} | MSE: {mean_squared_error(Y_test, Y_pred):.2f} | R²: {r2_score(Y_test, Y_pred):.2f}")

model("housing.csv", "total_rooms", "median_house_value", "Linear Regression - Housing")
model("auto-mpg.csv", "displacement", "mpg", "Polynomial Regression - Auto MPG", poly=True)
