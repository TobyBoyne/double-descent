"""Generate datapoints from a noisy step"""
import numpy as np

def noisy_step(seed=42, noise_variance=0.25):
    x = np.concatenate((
        np.linspace(-1.0, 0.0, num=16),
        np.linspace(0.5, 1.0, num=8),
    )).reshape(-1, 1)
    # x = np.linspace(-1., 1., num=10)[:, None]

    rng = np.random.default_rng(seed)
    e = rng.normal(loc=0, scale=noise_variance**0.5, size=x.shape)
    fx = np.where(x < -0.5, -1.0, 1.0)
    # fx = x
    y = fx + e
    return x, y