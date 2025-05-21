import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Olivetti faces dataset
faces = fetch_olivetti_faces()
X = faces.data
y = faces.target

# Function to display sample faces


def display_sample_faces(X, y, num_samples=5):
    """Display sample faces from the dataset"""
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.imshow(X[i].reshape(64, 64), cmap='gray')
        ax.set_title(f'Person {y[i]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize and train the Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Make predictions
y_pred = nb_classifier.predict(X_test)

# Calculate accuracy
accuracy = nb_classifier.score(X_test, y_test)

# Perform cross-validation
cv_scores = cross_val_score(nb_classifier, X, y, cv=5)

# Print performance metrics
print("Performance Metrics:")
print(f"\nAccuracy on test set: {accuracy:.4f}")
print("\nCross-validation scores:")
for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {score:.4f}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create confusion matrix visualization
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Function to test the classifier on specific samples


def test_specific_samples(classifier, X_test, y_test, num_samples=5):
    """Test the classifier on specific samples and display results"""
    # Randomly select samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    X_samples = X_test[indices]
    y_true = y_test[indices]

    # Make predictions
    y_pred = classifier.predict(X_samples)
    probabilities = classifier.predict_proba(X_samples)

    # Display results
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    for i in range(num_samples):
        # Display the face
        axes[0, i].imshow(X_samples[i].reshape(64, 64), cmap='gray')
        axes[0, i].axis('off')

        # Display prediction information
        axes[1, i].axis('off')
        prediction_text = f'True: {y_true[i]}\nPred: {y_pred[i]}\n'
        prediction_text += f'Prob: {probabilities[i][y_pred[i]]:.2f}'
        axes[1, i].text(0.5, 0.5, prediction_text,
                        ha='center', va='center')

        # Add color coding for correct/incorrect predictions
        if y_true[i] == y_pred[i]:
            axes[0, i].set_title('Correct', color='green')
        else:
            axes[0, i].set_title('Incorrect', color='red')

    plt.tight_layout()
    plt.show()


# Display sample faces from the dataset
print("\nDisplaying sample faces from the dataset:")
display_sample_faces(X, y)

# Test the classifier on specific samples
print("\nTesting classifier on specific samples:")
test_specific_samples(nb_classifier, X_test, y_test)

# Function to analyze misclassifications


def analyze_misclassifications(X_test, y_test, y_pred):
    """Analyze and display misclassified samples"""
    misclassified = X_test[y_test != y_pred]
    true_labels = y_test[y_test != y_pred]
    pred_labels = y_pred[y_test != y_pred]

    print(f"\nTotal misclassifications: {len(misclassified)}")

    # Display some misclassified examples
    num_display = min(5, len(misclassified))
    if num_display > 0:
        fig, axes = plt.subplots(1, num_display, figsize=(12, 3))
        for i in range(num_display):
            if num_display == 1:
                ax = axes
            else:
                ax = axes[i]
            ax.imshow(misclassified[i].reshape(64, 64), cmap='gray')
            ax.set_title(f'True: {true_labels[i]}\nPred: {pred_labels[i]}')
            ax.axis('off')
        plt.tight_layout()
        plt.show()


# Analyze misclassifications
print("\nAnalyzing misclassifications:")
analyze_misclassifications(X_test, y_test, y_pred)
