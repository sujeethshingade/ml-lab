import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_california_housing

# Load dataset
df = pd.read_csv('housing.csv')

# data = fetch_california_housing(as_frame=True)
# df = data.frame

# Select all numerical features
num_cols = df.select_dtypes(include=[np.number]).columns

# Plot histograms
n = len(num_cols)
rows = (n + 2) // 3
plt.figure(figsize=(15, 5 * rows))
for i, col in enumerate(num_cols, 1):
    plt.subplot(rows, 3, i)
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
plt.show()

# Plot box plots
plt.figure(figsize=(15, 5 * rows))
for i, col in enumerate(num_cols, 1):
    plt.subplot(rows, 3, i)
    sns.boxplot(x=df[col], color='orange')
    plt.title(f'Box Plot of {col}')
plt.show()

# Outlier Detection using IQR
print("\nOutlier Detection:")
outlier_summary = {}
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    count = len(outliers)
    outlier_summary[col] = count
    print(f"{col}: {count} outliers")

# Summary statistics
print(df.describe())
