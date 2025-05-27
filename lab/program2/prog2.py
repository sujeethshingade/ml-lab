import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Function to load and prepare the California housing dataset
def load_and_prepare_data():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['PRICE'] = housing.target
    return df

# Function to compute the correlation matrix for the dataframe
def compute_correlation_matrix(df):
    correlation_matrix = df.corr()
    return correlation_matrix

# Function to plot the correlation heatmap
def plot_correlation_heatmap(correlation_matrix):
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix,
                annot=True, # Show correlation values
                cmap='coolwarm', # Red for positive, blue for negative correlations
                vmin=-1, vmax=1, # Fix the range of correlation values
                center=0, # Center the colormap at 0
                square=True, # Make the plot square-shaped
                fmt='.2f') # Round correlation values to 2 decimal places
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.show()

# Function to create pair plot of the dataframe
def create_pair_plot(df):
    sns.pairplot(df, diag_kind='kde', plot_kws={'alpha': 0.6})
    plt.tight_layout()
    plt.show()

# Function to analyze strong correlations (|correlation| > 0.5) from the correlation matrix
def analyze_correlations(correlation_matrix):
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    strong_correlations = []
    for col in upper_tri.columns:
        for idx, value in upper_tri[col].items():
            if value is not None and abs(value) > 0.5:
                strong_correlations.append({
                    'features': (idx, col),
                    'correlation': value
                })
    strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    return strong_correlations

def main():
    print("Loading California Housing dataset...") # Load the data
    df = load_and_prepare_data()
 
    print("\nComputing correlation matrix...") # Compute correlation matrix
    correlation_matrix = compute_correlation_matrix(df)

    print("\nCreating correlation heatmap...") # Plot correlation heatmap
    plot_correlation_heatmap(correlation_matrix)

    print("\nCreating pair plot (this may take a moment)...") # Create pair plot
    create_pair_plot(df)

    print("\nAnalyzing strong correlations...") # Analyze and print notable correlations
    strong_correlations = analyze_correlations(correlation_matrix)

    print("\nStrong correlations found (|correlation| > 0.5):") # Print results
    for corr in strong_correlations:
        feature1, feature2 = corr['features']
        correlation = corr['correlation']
        correlation_type = "positive" if correlation > 0 else "negative"
        print(f"{feature1} and {feature2}: {correlation:.3f} ({correlation_type} correlation)")

# Calling the main function to execute the program
if __name__ == "__main__":
    main()
