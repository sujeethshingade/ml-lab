import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset
# df = fetch_california_housing(as_frame=True).frame

df = pd.read_csv('housing.csv')

# Compute the correlation matrix
corr = df.select_dtypes(include=[np.number]).corr()

# Plot heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of California Housing Features')
plt.tight_layout()
plt.show()

# Plot pairwise relationships using pairplot
sns.pairplot(df, diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle('Pair Plot of California Housing Features', y=1.02)
plt.tight_layout()
plt.show()
