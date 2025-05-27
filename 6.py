import numpy as np
import matplotlib.pyplot as plt

def gk(x, xi, tau): return np.exp(-np.sum((x - xi)**2) / (2*tau**2))

def lwr(x, X, y, tau):
    w = np.array([gk(x, X[i], tau) for i in range(len(X))])
    W = np.diag(w)
    return x @ np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y

np.random.seed(42)
X = np.linspace(0, 2*np.pi, 100)
y = np.sin(X) + 0.1*np.random.randn(100)
Xb = np.c_[np.ones(X.size), X]

xt = np.c_[np.ones(200), np.linspace(0, 2*np.pi, 200)]
tau = 0.5
yp = np.array([lwr(xi, Xb, y, tau) for xi in xt])

plt.scatter(X, y, c='r', alpha=0.7)
plt.plot(xt[:,1], yp, c='b')
plt.show()
