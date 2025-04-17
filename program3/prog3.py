import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_and_prepare_data():
    """Load Iris dataset and prepare it for PCA"""
    # Load the iris dataset
    iris = load_iris()

    # Create a DataFrame with feature names
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    # Add target variable
    df['target'] = iris.target
    df['target_names'] = pd.Categorical.from_codes(
        iris.target, iris.target_names)

    return df, iris.feature_names


def perform_pca(data, feature_names):
    """Perform PCA on the dataset"""
    # Separate features
    X = data[feature_names]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Calculate explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # Get component loadings
    loadings = pca.components_

    return X_pca, explained_variance_ratio, loadings, pca


def plot_pca_results(X_pca, data, explained_variance_ratio):
    """Plot the PCA results"""
    # Create figure
    plt.figure(figsize=(10, 8))

    # Create scatter plot for each class
    targets = sorted(data['target'].unique())
    target_names = sorted(data['target_names'].unique())

    for target, target_name in zip(targets, target_names):
        mask = data['target'] == target
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    label=target_name, alpha=0.8)

    # Add labels and title
    plt.xlabel(
        f'First Principal Component (Explains {explained_variance_ratio[0]:.2%} of variance)')
    plt.ylabel(
        f'Second Principal Component (Explains {explained_variance_ratio[1]:.2%} of variance)')
    plt.title('PCA of Iris Dataset')

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_explained_variance(pca):
    """Plot cumulative explained variance ratio"""
    plt.figure(figsize=(10, 6))
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumsum) + 1), cumsum, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance vs. Number of Components')
    plt.grid(True, alpha=0.3)
    plt.show()


def visualize_feature_importance(loadings, feature_names):
    """Visualize feature importance in each principal component"""
    plt.figure(figsize=(12, 6))

    # Plot for PC1
    plt.subplot(1, 2, 1)
    plt.bar(feature_names, loadings[0])
    plt.title('Feature Weights in First Principal Component')
    plt.xticks(rotation=45)

    # Plot for PC2
    plt.subplot(1, 2, 2)
    plt.bar(feature_names, loadings[1])
    plt.title('Feature Weights in Second Principal Component')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def main():
    # Load and prepare data
    print("Loading Iris dataset...")
    data, feature_names = load_and_prepare_data()

    # Perform PCA
    print("\nPerforming PCA...")
    X_pca, explained_variance_ratio, loadings, pca = perform_pca(
        data, feature_names)

    # Print explained variance
    print("\nExplained Variance Ratio:")
    print(f"PC1: {explained_variance_ratio[0]:.2%}")
    print(f"PC2: {explained_variance_ratio[1]:.2%}")
    print(f"Total: {sum(explained_variance_ratio):.2%}")

    # Plot results
    print("\nCreating visualizations...")
    plot_pca_results(X_pca, data, explained_variance_ratio)
    plot_explained_variance(pca)
    visualize_feature_importance(loadings, feature_names)

    # Print feature importance
    print("\nFeature Weights in Principal Components:")
    for i, component in enumerate(loadings):
        print(f"\nPrincipal Component {i+1}:")
        for fname, weight in zip(feature_names, component):
            print(f"{fname}: {weight:.3f}")


if __name__ == "__main__":
    main()
