import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from sklearn.datasets import load_iris

# Load Iris dataset and apply PCA
# iris = load_iris()

# Load your local CSV
df = pd.read_csv('IRIS.csv')

# Separate features and label column
label_column = df.columns[-1]
features = df.iloc[:, :-1]
labels = df[label_column]

# Apply PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(features)

# Create DataFrame for PCA results
pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
pca_df['Label'] = labels

# Plot the PCA results
plt.figure(figsize=(8, 6))
for label in pca_df['Label'].unique():
    plt.scatter(*pca_df[pca_df['Label'] == label]
                [['PC1', 'PC2']].T.values, label=label)

plt.title('PCA on Local CSV Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()
plt.show()
