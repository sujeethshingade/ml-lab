import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Wisconsin Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target  # We'll use this only for evaluation

# Create a DataFrame with feature names
df = pd.DataFrame(X, columns=data.feature_names)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to determine optimal k using elbow method


def plot_elbow_curve(X, max_k=10):
    inertias = []
    silhouette_scores = []
    k_values = range(2, max_k + 1)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    # Plot elbow curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Inertia plot
    ax1.plot(k_values, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')

    # Silhouette score plot
    ax2.plot(k_values, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')

    plt.tight_layout()
    plt.show()

    return k_values[np.argmax(silhouette_scores)]


# Find optimal k
optimal_k = plot_elbow_curve(X_scaled)
print(f"\nOptimal number of clusters based on silhouette score: {optimal_k}")

# Perform k-means clustering with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Perform PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create visualization of clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=cluster_labels, cmap='viridis')
plt.title('K-means Clustering Results (PCA-reduced data)')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.colorbar(scatter, label='Cluster')
plt.show()

# Compare clustering results with actual diagnosis
comparison_df = pd.DataFrame({
    'Cluster': cluster_labels,
    'Actual_Diagnosis': y
})
print("\nCluster vs Actual Diagnosis Distribution:")
print(pd.crosstab(comparison_df['Cluster'], comparison_df['Actual_Diagnosis'],
                  values=np.zeros_like(cluster_labels), aggfunc='count'))

# Analyze cluster characteristics


def analyze_clusters(X, labels, feature_names):
    """Analyze and visualize characteristics of each cluster"""
    # Create DataFrame with features and cluster labels
    df_analysis = pd.DataFrame(X, columns=feature_names)
    df_analysis['Cluster'] = labels

    # Calculate mean values for each feature in each cluster
    cluster_means = df_analysis.groupby('Cluster').mean()

    # Create heatmap of cluster characteristics
    plt.figure(figsize=(15, 8))
    sns.heatmap(cluster_means, cmap='coolwarm', center=0, annot=True, fmt='.2f',
                xticklabels=True, yticklabels=True)
    plt.title('Cluster Characteristics (Feature Means)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return cluster_means


# Analyze cluster characteristics
print("\nAnalyzing cluster characteristics:")
cluster_means = analyze_clusters(X_scaled, cluster_labels, data.feature_names)

# Visualize feature importance for clustering


def plot_feature_importance(kmeans, feature_names):
    """Plot feature importance based on cluster centroids"""
    # Calculate the variance of centroids for each feature
    centroid_variance = np.var(kmeans.cluster_centers_, axis=0)

    # Create DataFrame for feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': centroid_variance
    }).sort_values('Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Top 10 Most Important Features for Clustering')
    plt.tight_layout()
    plt.show()

    return feature_importance


# Plot feature importance
print("\nAnalyzing feature importance:")
feature_importance = plot_feature_importance(kmeans, data.feature_names)

# Function to predict cluster for new samples


def predict_cluster(sample, scaler, kmeans, feature_names):
    """Predict cluster for a new sample"""
    # Ensure sample is in correct format
    if isinstance(sample, list):
        sample = np.array(sample).reshape(1, -1)

    # Scale the sample
    sample_scaled = scaler.transform(sample)

    # Predict cluster
    cluster = kmeans.predict(sample_scaled)[0]

    # Get distances to all cluster centers
    distances = kmeans.transform(sample_scaled)[0]

    print(f"\nPredicted Cluster: {cluster}")
    print("\nDistances to cluster centers:")
    for i, dist in enumerate(distances):
        print(f"Cluster {i}: {dist:.2f}")

    return cluster, distances


# Example of using the prediction function
print("\nExample prediction for a new sample:")
example_sample = X[0:1]  # Using first sample as example
predicted_cluster, distances = predict_cluster(
    example_sample, scaler, kmeans, data.feature_names)
