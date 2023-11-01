"""A Fourier model"""
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

import gpflow
import tensorflow as tf
from doubledescent.fourierkernel import Fourier
from doubledescent.step import noisy_step

NOISE_VAR = 0.25

def fit_fourier_model(X, Y, degree: int, scaling: int):
    # log_marginal_likelihood
    # define model
    model = gpflow.models.GPR(
        (X, Y),
        kernel=Fourier(degree=degree, scaling=scaling),
        noise_variance=NOISE_VAR
    )

    gpflow.set_trainable(model.likelihood, False)
    gpflow.set_trainable(model.kernel.lengthscales, False)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)
    
    return model

def plot_fitted_model(X, Y, model, ax: Axes):
    Xplot = np.linspace(-1.8, 1.8, 100)[:, None]
    f_mean, _ = model.predict_f(Xplot, full_cov=False)

    # have to manually compute the variance due to instability in gpflow
    kernel: Fourier = model.kernel
    Kss = kernel(Xplot, Xplot)
    Ksx = kernel(Xplot, X)
    Kxx = kernel(X, X)
    Kxs = kernel(X, Xplot)
    var_mat = Kss - Ksx @ (tf.linalg.inv(Kxx + NOISE_VAR * tf.eye(X.shape[0], dtype=tf.float64)) @ Kxs)
    f_var = tf.linalg.diag_part(var_mat)[:, None]

    f_lower = f_mean - 1.96 * np.sqrt(f_var)
    f_upper = f_mean + 1.96 * np.sqrt(f_var)

    ax.scatter(X, Y, color="black", marker="x")
    (mean_line,) = ax.plot(Xplot, f_mean, "-", label=kernel.__class__.__name__)
    color = mean_line.get_color()
    ax.plot(Xplot, f_lower, lw=0.1, color=color)
    ax.plot(Xplot, f_upper, lw=0.1, color=color)
    ax.fill_between(
        Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color=color, alpha=0.3
    )
    ax.set_ylim(bottom=-3.0, top=3.0)
    # ax.set_title(f"GP fit to data")
    ax.set_title(f"degree={kernel.degree}")

def plot_kernel_samples(model, ax: Axes) -> None:
    Xplot = np.linspace(-1.8, 1.8, 100)[:, None]
    tf.random.set_seed(20220903)
    n_samples = 10
    # predict_f_samples draws n_samples examples of the function f, and returns their values at Xplot.
    fs = model.predict_f_samples(Xplot, n_samples)
    ax.plot(Xplot, fs[:, :, 0].numpy().T, label=model.kernel.__class__.__name__)
    ax.set_ylim(bottom=-4.0, top=4.0)
    ax.set_title("Sample functions from prior")

def main_increasing_degree(scaling):
    X, Y = noisy_step(noise_variance=NOISE_VAR)
    fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(20, 10))
    degs = 2 * np.arange(axs.size)
    marginal_likelihoods = []
    for ax, deg in zip(axs.flatten(), degs):
        model = fit_fourier_model(X, Y, deg, scaling)
        plot_fitted_model(X, Y, model, ax)
        marginal_likelihoods.append(model.log_marginal_likelihood())
    fig_marginals, ax_marginals = plt.subplots(figsize=(6, 1.5))
    ax_marginals: Axes
    probs = tf.exp(tf.stack(marginal_likelihoods))
    ax_marginals.bar(degs, probs / tf.reduce_sum(probs), width=1.6)
    ax_marginals.set_xticks(degs)
    ax_marginals.set_xlabel("Model degree")
    ax_marginals.set_ylabel("Model evidence,\n$p(\mathcal{M} | \mathcal{D})$")
    
    fig.savefig(f"figs/fit_kernels_{scaling}.pdf")
    fig_marginals.savefig(f"figs/model_probs_{scaling}.pdf", bbox_inches="tight")


def main_single_figs():
    """Plot many single figures for demonstration"""
    X, Y = noisy_step(noise_variance=NOISE_VAR)
    degs = [1, 2, 4, 12, 20]
    scaling = 0
    for deg in degs:
        model = fit_fourier_model(X, Y, deg, scaling)
        fig, (ax_prior, ax_fit) = plt.subplots(ncols=2, figsize=(12, 6))
        plot_kernel_samples(model, ax_prior)
        plot_fitted_model(X, Y, model, ax_fit)
        fig.savefig(f"figs/single/fit_deg{deg}.pdf")


if __name__ == "__main__":
    main_increasing_degree(scaling=0)
    main_increasing_degree(scaling=3)
    main_increasing_degree(scaling=6)