import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Set the style for better visualization
plt.style.use('tableau-colorblind10')  # Using a built-in matplotlib style

def load_and_prepare_data():
    """Load California Housing dataset and convert to pandas DataFrame"""
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['PRICE'] = housing.target
    return df

def create_distribution_plots(df, save_plots=False):
    """Create histograms and box plots for all numerical features"""
    numerical_features = df.columns

    # Calculate number of rows needed for subplot grid
    n_features = len(numerical_features)
    n_rows = (n_features + 1) // 2  # 2 plots per row

    # Create histograms
    plt.figure(figsize=(15, 5*n_rows))
    for idx, feature in enumerate(numerical_features, 1):
        plt.subplot(n_rows, 2, idx)
        sns.histplot(data=df, x=feature, kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
    plt.tight_layout()
    if save_plots:
        plt.savefig('histograms.png')
    plt.show()

    # Create box plots
    plt.figure(figsize=(15, 5*n_rows))
    for idx, feature in enumerate(numerical_features, 1):
        plt.subplot(n_rows, 2, idx)
        sns.boxplot(data=df[feature])
        plt.title(f'Box Plot of {feature}')
    plt.tight_layout()
    if save_plots:
        plt.savefig('boxplots.png')
    plt.show()

def analyze_distributions(df):
    """Generate statistical summary and identify outliers"""
    stats_summary = df.describe()

    # Calculate IQR and identify outliers for each feature
    outlier_summary = {}
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]

        outlier_summary[column] = {
            'number_of_outliers': len(outliers),
            'percentage_of_outliers': (len(outliers) / len(df)) * 100,
            'outlier_range': f"< {lower_bound:.2f} or > {upper_bound:.2f}"
        }

    return stats_summary, outlier_summary

def main():
    # Load the data
    df = load_and_prepare_data()

    # Create visualization plots
    create_distribution_plots(df)

    # Analyze distributions and outliers
    stats_summary, outlier_summary = analyze_distributions(df)

    # Print statistical summary
    print("\nStatistical Summary:")
    print(stats_summary)

    # Print outlier analysis
    print("\nOutlier Analysis:")
    for feature, summary in outlier_summary.items():
        print(f"\n{feature}:")
        print(f"Number of outliers: {summary['number_of_outliers']}")
        print(f"Percentage of outliers: {summary['percentage_of_outliers']:.2f}%")
        print(f"Outlier range: {summary['outlier_range']}")

if __name__ == "__main__":
    main()
