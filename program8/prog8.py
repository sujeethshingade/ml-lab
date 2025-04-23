from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create and train the decision tree classifier
dt_classifier = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Print model performance metrics
print("Model Performance Metrics:")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Malignant', 'Benign']))

# Create confusion matrix visualization
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, feature_names=data.feature_names,
          class_names=['Malignant', 'Benign'], filled=True, rounded=True)
plt.title('Decision Tree Visualization')
plt.show()

# Function to classify a new sample


def classify_new_sample(sample, feature_names=data.feature_names):
    """
    Classify a new sample using the trained decision tree model.

    Parameters:
    sample (list or array): List of feature values in the same order as the training data
    feature_names (list): List of feature names for reference

    Returns:
    tuple: (prediction, probability)
    """
    sample = np.array(sample).reshape(1, -1)
    prediction = dt_classifier.predict(sample)
    probability = dt_classifier.predict_proba(sample)

    print("\nClassification Results:")
    print(f"Prediction: {'Benign' if prediction[0] == 1 else 'Malignant'}")
    print(
        f"Probability: Malignant: {probability[0][0]:.2f}, Benign: {probability[0][1]:.2f}")

    # Print feature importance for this prediction
    print("\nTop 5 Most Important Features:")
    importances = dict(zip(feature_names, dt_classifier.feature_importances_))
    sorted_importances = sorted(
        importances.items(), key=lambda x: x[1], reverse=True)[:5]
    for feature, importance in sorted_importances:
        print(f"{feature}: {importance:.4f}")

    return prediction[0], probability[0]


# Example of using the classifier with a new sample
# Using mean values from the dataset as an example
example_sample = X_train.mean(axis=0)
print("\nExample Classification:")
classify_new_sample(example_sample)
