"""Implement the sample-then-optimise process described in [Matthews et. al, 2017]"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def objective(Phi, y):
    def inner(w):
        e = Phi @ w[:, None] - y
        # return objective and gradient
        
        return (e.T @ e, 2 * Phi.T @ e)
    return inner

def phi(X: np.ndarray, c: np.ndarray, l: float = 0.5):
    # X: [N x 1]
    # c: [1 x M]
    # Phi: [N x M]

    r = X - c
    return np.exp(-(r / l)**2)

M = 50
N = 6
X = np.linspace(-2, 2, num=N)[:, None]
y = np.array([-0.1, 0.1, 0.0, 1.5, 1.3, -0.5])[:, None]
C = np.linspace(-10, 10, num=M)[None, :]
Phi = phi(X, C)
rng = np.random.default_rng(seed=42)
w = rng.normal(loc=0, scale=1, size=(M, 1))

# [N x M] @ [M x 1] = [N x 1]
print(Phi.shape)
f = Phi @ w
minresult = minimize(objective(Phi, y), x0=w.flatten(), jac=True, method="L-BFGS-B")
w = minresult.x[:, None]

Xplot = np.linspace(-3, 3, num=100)[:, None]
f = phi(Xplot, C) @ w

plt.plot(Xplot, f)
plt.scatter(X, y)
plt.show()