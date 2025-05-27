import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

# Load and prepare data
df = pd.read_csv('breast_cancer.csv')
X = df.select_dtypes(include=['float64', 'int64'])  # numeric features
y = df['Status'].map({'Alive': 0, 'Dead': 1})       # convert to numeric labels

# Scale and cluster
X_scaled = StandardScaler().fit_transform(X)
labels = KMeans(n_clusters=2, random_state=42).fit_predict(X_scaled)

# Evaluate
print(confusion_matrix(y, labels))
print(classification_report(y, labels))

# Visualize
pca = PCA(n_components=2).fit_transform(X_scaled)
sns.scatterplot(x=pca[:, 0], y=pca[:, 1], hue=labels, palette='Set1')
plt.title('K-Means Clustering'); plt.xlabel('PC1'); plt.ylabel('PC2')
plt.show()
