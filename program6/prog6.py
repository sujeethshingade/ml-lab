import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


def generate_sample_data(n_samples=100, noise=10):
    """Generate sample data with non-linear pattern"""
    X = np.linspace(0, 10, n_samples)
    y = 2 * np.sin(X) + X/2 + np.random.normal(0, noise/10, n_samples)
    return X, y


def kernel(x, x_i, tau=0.5):
    """Gaussian kernel function for weight calculation"""
    return np.exp(-(x - x_i)**2 / (2 * tau**2))


def lowess(X, y, x_pred, tau=0.5):
    """
    Locally Weighted Regression implementation

    Parameters:
    -----------
    X : array-like
        Training input features
    y : array-like
        Target values
    x_pred : array-like
        Points at which to make predictions
    tau : float
        Bandwidth parameter controlling smoothness

    Returns:
    --------
    array-like
        Predicted values at x_pred points
    """
    # Ensure arrays are 1D
    X = np.ravel(X)
    y = np.ravel(y)
    x_pred = np.ravel(x_pred)

    y_pred = []

    for x in x_pred:
        # Calculate weights for all points
        weights = kernel(x, X, tau)

        # Weighted least squares matrices
        W = np.diag(weights)
        X_aug = np.column_stack([np.ones_like(X), X])  # Add bias term

        # Calculate weighted least squares parameters
        theta = np.linalg.inv(X_aug.T @ W @ X_aug) @ X_aug.T @ W @ y

        # Make prediction
        x_aug = np.array([1, x])
        y_pred.append(float(x_aug @ theta))

    return np.array(y_pred)


# Generate sample data
np.random.seed(42)
X, y = generate_sample_data(n_samples=100, noise=10)

# Generate points for prediction
X_pred = np.linspace(0, 10, 200)

# Fit LOWESS with different bandwidth parameters
y_pred_smooth = lowess(X, y, X_pred, tau=0.3)  # More local fitting
y_pred_medium = lowess(X, y, X_pred, tau=0.8)  # Medium smoothing
y_pred_rough = lowess(X, y, X_pred, tau=2.0)   # More global fitting

# Plotting
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Data points')
plt.plot(X_pred, y_pred_smooth, 'r-',
         label='τ = 0.3 (More local)', linewidth=2)
plt.plot(X_pred, y_pred_medium, 'g-', label='τ = 0.8 (Medium)', linewidth=2)
plt.plot(X_pred, y_pred_rough, 'y-',
         label='τ = 2.0 (More global)', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Locally Weighted Regression with Different Bandwidth Parameters')
plt.legend()
plt.grid(True)
plt.show()
