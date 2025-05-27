import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

data = np.random.rand(100)
train, test = data[:50], data[50:]
labels = ["Class1" if x <= 0.5 else "Class2" for x in train]

def knn(train, labels, point, k):
    dist = sorted([(abs(point - train[i]), labels[i]) for i in range(len(train))])
    return Counter([label for _, label in dist[:k]]).most_common(1)[0][0]

for k in [1, 2, 3, 4, 5, 20, 30]:
    preds = [knn(train, labels, x, k) for x in test]
    c1 = [x for i, x in enumerate(test) if preds[i] == "Class1"]
    c2 = [x for i, x in enumerate(test) if preds[i] == "Class2"]

    plt.scatter(train, [0]*50, c=["blue" if l=="Class1" else "red" for l in labels], marker='o')
    plt.scatter(c1, [1]*len(c1), c='blue', marker='x')
    plt.scatter(c2, [1]*len(c2), c='red', marker='x')
    plt.title(f'k = {k}')
    plt.yticks([0, 1], ['Train', 'Test'])
    plt.xlabel('Value')
    plt.grid()
    plt.show()
