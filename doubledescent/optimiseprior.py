"""Implement the sample-then-optimise process described in [Matthews et. al, 2017]"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Callable

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
    return np.exp(- (r / l)**2)

def sample_from_prior(M: int, num_samples: int, seed=42):
    """Returns function evaluations [N x num_samples]"""
    rng = np.random.default_rng(seed=seed)
    return rng.normal(loc=0, scale=1, size=(M, num_samples))

def optimise_priors(ws: np.ndarray, objective: Callable):
    # ws : [M x num_samples]
    ws_opt = np.empty_like(ws)
    for i, w in enumerate(np.moveaxis(ws, -1, 0)):
        res = minimize(objective, x0=w.flatten(), jac=True, method="L-BFGS-B")
        ws_opt[:, i] = res.x
    
    return ws_opt
