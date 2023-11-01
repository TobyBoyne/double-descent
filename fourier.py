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
        Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color=color, alpha=0.1
    )
    ax.set_ylim(bottom=-3.0, top=3.0)
    ax.set_title(f"degree={kernel.degree}")



def main_increasing_degree():
    X, Y = noisy_step(noise_variance=NOISE_VAR)
    fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(20, 10))
    degs = 2 * np.arange(axs.size)
    scaling = 0
    # degs[-1] = 500
    marginal_likelihoods = []
    for ax, deg in zip(axs.flatten(), degs):
        model = fit_fourier_model(X, Y, deg, scaling)
        plot_fitted_model(X, Y, model, ax)
        marginal_likelihoods.append(model.log_marginal_likelihood())
    fig_marginals, ax_marginals = plt.subplots()
    ax_marginals: Axes
    probs = tf.exp(tf.stack(marginal_likelihoods))
    ax_marginals.bar(degs, probs / tf.reduce_sum(probs), width=1.6)
    ax_marginals.set_xticks(degs)
    ax_marginals.set_xlabel("Model degree")
    ax_marginals.set_ylabel("Model probability, $p(\mathcal{M} | \mathcal{D})$")
    
    fig.savefig(f"figs/fit_kernels_{scaling}.pdf")
    fig_marginals.savefig(f"figs/model_probs_{scaling}.pdf")
    plt.show()

if __name__ == "__main__":
    main_increasing_degree()