import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("Breast_Cancer.csv")

# Convert text columns to numbers (except target)
for col in df.select_dtypes(include='object').columns:
    if col != 'Status':
        df[col] = LabelEncoder().fit_transform(df[col])

# Split features and target
X = df.drop('Status', axis=1)
y = df['Status']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Check accuracy
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)*100:.2f}%")

# Predict first test sample
print("Prediction for first test sample:", predictions[0])

# Visualize tree
plt.figure(figsize=(12,8))
plot_tree(model, filled=True, feature_names=X.columns)
plt.show()
