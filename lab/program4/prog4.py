import pandas as pd
import numpy as np


class FindS:
    def __init__(self):
        self.hypothesis = None
        self.features = None

    def initialize_hypothesis(self, num_features):
        """Initialize the most specific hypothesis"""
        return ['ϕ'] * num_features

    def is_positive_example(self, target):
        """Check if the example is positive"""
        return target == 'Yes'

    def generalize_hypothesis(self, example, current_hypothesis):
        """
        Generalize the hypothesis to be consistent with the positive example
        """
        new_hypothesis = []

        for ex_val, hyp_val in zip(example, current_hypothesis):
            # If hypothesis value is 'ϕ' (null), use the example value
            if hyp_val == 'ϕ':
                new_hypothesis.append(ex_val)
            # If values match, keep the value
            elif ex_val == hyp_val:
                new_hypothesis.append(hyp_val)
            # If values don't match, generalize to '?'
            else:
                new_hypothesis.append('?')

        return new_hypothesis

    def fit(self, data, target_column):
        """
        Find the most specific hypothesis consistent with the training examples

        Parameters:
        data: pandas DataFrame containing the training examples
        target_column: name of the target column
        """
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Store feature names
        self.features = X.columns.tolist()

        # Initialize hypothesis
        self.hypothesis = self.initialize_hypothesis(len(self.features))

        # Process each training example
        for index, row in X.iterrows():
            # Only consider positive examples
            if self.is_positive_example(y[index]):
                self.hypothesis = self.generalize_hypothesis(
                    row.values.tolist(),
                    self.hypothesis
                )

        return self.hypothesis

    def print_hypothesis(self):
        """Print the current hypothesis in a readable format"""
        if self.hypothesis and self.features:
            print("\nFinal Hypothesis:")
            print("〈", end='')
            for feature, value in zip(self.features, self.hypothesis):
                print(f"{feature} = {value}, ", end='')
            print("〉")
        else:
            print("No hypothesis found. Please run fit() first.")


def load_data(filename):
    """Load data from CSV file"""
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None


def main():
    # Example usage with sample data
    print("Creating sample training data...")

    # Create sample data if no file is provided
    sample_data = {
        'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
        'Temperature': ['Warm', 'Warm', 'Cold', 'Warm'],
        'Humidity': ['High', 'High', 'High', 'High'],
        'Wind': ['Weak', 'Strong', 'Weak', 'Weak'],
        'PlayTennis': ['Yes', 'Yes', 'No', 'Yes']
    }

    df = pd.DataFrame(sample_data)
    print("\nTraining Data:")
    print(df)

    # Initialize and run Find-S algorithm
    print("\nRunning Find-S algorithm...")
    find_s = FindS()
    find_s.fit(df, target_column='PlayTennis')

    # Print results
    find_s.print_hypothesis()

    print("\nHypothesis Interpretation:")
    print("- '?' means any value is acceptable for that attribute")
    print("- 'ϕ' means no value has been observed (null)")
    print("- Specific values indicate required values for that attribute")


if __name__ == "__main__":
    main()
