import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load image data
X = np.load("olivetti_faces.npy")  # shape: (400, 64, 64)
X = X.reshape(400, -1)            # flatten to (400, 4096)
y = np.repeat(np.arange(40), 10)  # labels: 0–39, 10 images each

# Train/test split and model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = GaussianNB().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%')
print("\nReport:\n", classification_report(y_test, y_pred, zero_division=1))
print("\nConfusion:\n", confusion_matrix(y_test, y_pred))
print(f'\nCross-val Accuracy: {cross_val_score(model, X, y, cv=5).mean()*100:.2f}%')

# Plot some predictions
fig, axes = plt.subplots(3, 5, figsize=(12, 8))
for ax, img, true, pred in zip(axes.ravel(), X_test, y_test, y_pred):
    ax.imshow(img.reshape(64, 64), cmap='gray')
    ax.set_title(f"T:{true}, P:{pred}")
    ax.axis('off')
plt.tight_layout(); plt.show()
