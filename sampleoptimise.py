"""Implement the sample-then-optimise process described in [Matthews et. al, 2017]"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from scipy.optimize import minimize
from doubledescent.optimiseprior import optimise_priors, objective, sample_from_prior, phi



def main(M, N, num_samples, highlight_one=True):
    X = np.linspace(-2, 2, num=N)[:, None]
    # y = np.array([-0.1, 0.1, 0.0, 1.5, 1.3, -0.5])[:, None]

    y = np.random.default_rng(seed=42).uniform(low=-0.5, high=0.5, size=(N, 1))
    C = np.linspace(-3, 3, num=M)[None, :]

    Phi = phi(X, C)
    # [N x M] @ [M x 1] = [N x 1]
    ws = sample_from_prior(M, num_samples)
    ws_post = optimise_priors(ws, objective(Phi, y))

    Xplot = np.linspace(-2.5, 2.5, num=100)[:, None]
    Phi_plot = phi(Xplot, C)

    fs_prior = Phi_plot @ ws
    fs_post = Phi_plot @ ws_post
    fig, axs = plt.subplots(nrows=3, figsize=(10, 10))
    axs: list[Axes]

    cm = mpl.colormaps["viridis"]
    num_show = 5
    colors = [cm(i) for i in np.linspace(0, 1, num=num_show)]
    if highlight_one:
        colors = [(*c[:-1], 1.0 if i==1 else 0.2) for i, c in enumerate(colors)]
    axs[0].set_prop_cycle('color', colors)
    axs[0].plot(Xplot, fs_prior[:, :num_show], linewidth=3)
    axs[0].set_ylim(-3.0, 3.0)

    axs[1].set_prop_cycle('color', colors)
    axs[1].plot(Xplot, fs_post[:, :num_show], linewidth=3)
    axs[1].scatter(X, y, marker="+", s=100, linewidths=3, zorder=100, alpha=1.0)
    axs[1].set_ylim(-2.0, 2.0)

    f_mean = np.mean(fs_post, axis=-1, keepdims=True)
    f_var = np.var(fs_post, axis=-1, keepdims=True)

    f_lower = f_mean - 1.96 * np.sqrt(f_var)
    f_upper = f_mean + 1.96 * np.sqrt(f_var)
    (mean_line,) = axs[2].plot(Xplot, f_mean, "-")
    color = mean_line.get_color()
    axs[2].plot(Xplot, f_lower, lw=0.1, color=color)
    axs[2].plot(Xplot, f_upper, lw=0.1, color=color)
    axs[2].fill_between(
        Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color=color, alpha=0.3
    )
    axs[2].scatter(X, y, marker="+", color="black", s=100, linewidths=3, zorder=100)
    axs[1].set_ylim(-2.0, 2.0)


    axs[0].set_title("Prior samples", fontdict={"size":16})
    axs[1].set_title("Posterior samples (after optimisation)", fontdict={"size":16})
    axs[2].set_title("Mean and variance of posterior samples",  fontdict={"size":16})
    for ax in axs:
        ax.set_xlim(Xplot.min(), Xplot.max())

    fig.tight_layout()


    fig.savefig(f"figs/_sampleoptimise_underparametrised.pdf")
    # fig.savefig(f"figs/sampleoptimise{''if not highlight_one else '_highlight'}.pdf")


if __name__ == "__main__":
    main(M=5, N=10, num_samples=100, highlight_one=False)
    plt.show()