from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.axes import Axes
from matplotlib.cm import coolwarm

import gpflow
from gpflow.utilities import print_summary

from doubledescent.fourierkernel import Fourier
from doubledescent.step import noisy_step

def plot_kernel_samples(ax: Axes, kernel: gpflow.kernels.Kernel) -> None:
    X = np.zeros((0, 1))
    Y = np.zeros((0, 1))
    model = gpflow.models.GPR((X, Y), kernel=deepcopy(kernel))
    Xplot = np.linspace(-0.6, 0.6, 100)[:, None]
    tf.random.set_seed(20220903)
    n_samples = 10
    # predict_f_samples draws n_samples examples of the function f, and returns their values at Xplot.
    fs = model.predict_f_samples(Xplot, n_samples)
    ax.plot(Xplot, fs[:, :, 0].numpy().T, label=kernel.__class__.__name__)
    ax.set_ylim(bottom=-2.0, top=2.0)
    ax.set_title("Example $f$s")


def plot_kernel_prediction(
    ax: Axes, kernel: gpflow.kernels.Kernel, *, optimise: bool = True
) -> None:
    noise_var = 0.25
    X, Y = noisy_step()
    model = gpflow.models.GPR(
        (X, Y), kernel=deepcopy(kernel), noise_variance=noise_var
    )

    if optimise:
        gpflow.set_trainable(model.likelihood, False)
        gpflow.set_trainable(model.kernel.lengthscales, False)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)
        print_summary(model)
    Xplot = np.linspace(-1.5, 1.5, 100)[:, None]

    f_mean, f_var = model.predict_f(Xplot, full_cov=False)

    Kss = model.kernel(Xplot, Xplot)
    Ksx = model.kernel(Xplot, X)
    Kxx = model.kernel(X, X)
    Kxs = model.kernel(X, Xplot)
    my_var = Kss - Ksx @ (tf.linalg.inv(Kxx + noise_var * tf.eye(X.shape[0], dtype=tf.float64)) @ Kxs)
    f_var = tf.linalg.diag_part(my_var)[:, None]
    
    f_lower = f_mean - 1.96 * np.sqrt(f_var)
    f_upper = f_mean + 1.96 * np.sqrt(f_var)

    ax.scatter(X, Y, color="black")
    (mean_line,) = ax.plot(Xplot, f_mean, "-", label=kernel.__class__.__name__)
    color = mean_line.get_color()
    ax.plot(Xplot, f_lower, lw=0.1, color=color)
    ax.plot(Xplot, f_upper, lw=0.1, color=color)
    ax.fill_between(
        Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color=color, alpha=0.1
    )
    ax.set_ylim(bottom=-2.0, top=5.0)
    ax.set_title("Example data fit")


def plot_kernel(
    kernel: gpflow.kernels.Kernel, *, optimise: bool = True
) -> None:
    _, (samples_ax, prediction_ax) = plt.subplots(nrows=1, ncols=2)
    plot_kernel_samples(samples_ax, kernel)
    plot_kernel_prediction(prediction_ax, kernel, optimise=optimise)


plot_kernel(Fourier(degree=12), optimise=True)
# plot_kernel(gpflow.kernels.SquaredExponential())
plt.show()