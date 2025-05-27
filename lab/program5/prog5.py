import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class KNN:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store training data"""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predict class for each input value"""
        predictions = []

        for x in X:
            # Calculate distances to all training points
            distances = np.abs(self.X_train - x)

            # Get indices of k nearest neighbors
            k_nearest_indices = np.argsort(distances)[:self.k]

            # Get classes of k nearest neighbors
            k_nearest_labels = self.y_train[k_nearest_indices]

            # Perform majority voting
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])

        return np.array(predictions)


def generate_data():
    """Generate and label the dataset"""
    # Generate 100 random points in [0,1]
    np.random.seed(42)  # For reproducibility
    X = np.random.rand(100)

    # Label first 50 points
    y = np.zeros(100)
    y[:50] = np.where(X[:50] <= 0.5, 1, 2)

    return X, y


def plot_results(X_train, y_train, X_test, y_pred, k):
    """Plot the results for a given k value"""
    plt.figure(figsize=(12, 4))

    # Plot training data
    plt.scatter(X_train[y_train == 1], np.zeros_like(X_train[y_train == 1]),
                c='blue', label='Class 1 (Training)', marker='o')
    plt.scatter(X_train[y_train == 2], np.zeros_like(X_train[y_train == 2]),
                c='red', label='Class 2 (Training)', marker='o')

    # Plot test data predictions
    plt.scatter(X_test[y_pred == 1], np.ones_like(X_test[y_pred == 1])*0.1,
                c='lightblue', label='Class 1 (Predicted)', marker='^')
    plt.scatter(X_test[y_pred == 2], np.ones_like(X_test[y_pred == 2])*0.1,
                c='lightcoral', label='Class 2 (Predicted)', marker='^')

    plt.title(f'KNN Classification Results (k={k})')
    plt.xlabel('x')
    plt.yticks([])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def analyze_boundary_points(X_test, y_pred, k):
    """Analyze and print details about boundary points"""
    boundary_points = []

    # Find points where predictions change
    for i in range(1, len(y_pred)):
        if y_pred[i] != y_pred[i-1]:
            boundary_points.append(X_test[i])

    if boundary_points:
        print(f"\nDecision boundaries for k={k}:")
        for point in sorted(boundary_points):
            print(f"x = {point:.3f}")
    else:
        print(f"\nNo clear decision boundaries found for k={k}")


def main():
    # Generate data
    print("Generating dataset...")
    X, y = generate_data()

    # Split into training and test sets
    X_train, y_train = X[:50], y[:50]
    X_test, y_test = X[50:], y[50:]

    # Sort test data for better visualization
    sort_idx = np.argsort(X_test)
    X_test = X_test[sort_idx]

    # Try different k values
    k_values = [1, 2, 3, 4, 5, 20, 30]

    for k in k_values:
        print(f"\nPerforming classification with k={k}")

        # Create and train KNN classifier
        knn = KNN(k=k)
        knn.fit(X_train, y_train)

        # Make predictions
        y_pred = knn.predict(X_test)

        # Plot results
        plot_results(X_train, y_train, X_test, y_pred, k)

        # Analyze decision boundaries
        analyze_boundary_points(X_test, y_pred, k)

        # Calculate and print summary statistics
        class1_pred = np.sum(y_pred == 1)
        class2_pred = np.sum(y_pred == 2)
        print(f"\nPrediction Summary for k={k}:")
        print(
            f"Class 1: {class1_pred} points ({class1_pred/len(y_pred)*100:.1f}%)")
        print(
            f"Class 2: {class2_pred} points ({class2_pred/len(y_pred)*100:.1f}%)")


if __name__ == "__main__":
    main()
