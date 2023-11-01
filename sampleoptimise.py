"""Implement the sample-then-optimise process described in [Matthews et. al, 2017]"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from doubledescent.optimiseprior import optimise_priors, objective, sample_from_prior, phi

# M = 20
# N = 6
# X = np.linspace(-2, 2, num=N)[:, None]
# y = np.array([-0.1, 0.1, 0.0, 1.5, 1.3, -0.5])[:, None]
# C = np.linspace(-10, 10, num=M)[None, :]
# Phi = phi(X, C)
# rng = np.random.default_rng(seed=42)
# w = rng.normal(loc=0, scale=1, size=(M, 1))

# # [N x M] @ [M x 1] = [N x 1]
# print(Phi.shape)
# f = Phi @ w
# minresult = minimize(objective(Phi, y), x0=w.flatten(), jac=True, method="L-BFGS-B")
# w = minresult.x[:, None]

# Xplot = np.linspace(-2.5, 2.5, num=100)[:, None]
# f = phi(Xplot, C) @ w

# plt.plot(Xplot, f)
# plt.scatter(X, y)
# plt.show()

def main(M=20, num_samples=10):
    N = 6
    X = np.linspace(-2, 2, num=N)[:, None]
    y = np.array([-0.1, 0.1, 0.0, 1.5, 1.3, -0.5])[:, None]
    C = np.linspace(-10, 10, num=M)[None, :]

    Phi = phi(X, C)
    # [N x M] @ [M x 1] = [N x 1]
    ws = sample_from_prior(M, num_samples)
    fs_prior = Phi @ ws
    ws_post = optimise_priors(ws, objective(Phi, y))

    Xplot = np.linspace(-2.5, 2.5, num=100)[:, None]
    Phi_plot = phi(Xplot, C)
    fs_post = Phi_plot @ ws_post
    plt.plot(Xplot, fs_post)
    plt.scatter(X, y)
    plt.show()

if __name__ == "__main__":
    main(M=50, num_samples=100)