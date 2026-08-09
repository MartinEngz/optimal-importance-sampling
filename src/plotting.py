import numpy as np
import matplotlib.pyplot as plt

from src.statistics import standard_error, confidence_interval

def plot_standard_error_convergence(results, title):
    # rows in results: [N, estimate, std]
    N = np.array([res[0] for res in results], dtype=float)
    std = np.array([res[2] for res in results], dtype=float)

    se = standard_error(std, N)

    # Reference slope ~ 1/sqrt(N)
    # Anchor N^{-1/2} reference at first non-zero empirical SE
    positive_idx = np.flatnonzero(se > 0)[0]
    ref_se = se[positive_idx] * np.sqrt(N[positive_idx]) / np.sqrt(N)

    plt.figure(figsize=(8.5, 4.8))

    plt.loglog(
        N,
        se,
        "o-",
        linewidth=1.2,
        markersize=5,
        label="Empirical SE"
    )

    plt.loglog(
        N,
        ref_se,
        "--",
        linewidth=1.2,
        label=r"Reference $\propto N^{-1/2}$"
    )

    plt.xlabel("Number of samples (N)")
    plt.ylabel("Standard error")
    plt.title(title)
    plt.grid(True, which='both', linestyle=':', linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_estimates_with_ci(
    results,
    reference=None,
    reference_label=None,
    title=None,
    ylabel="Option Price Estimate",
    alpha=0.05
):
    N = np.array([res[0] for res in results], dtype=float)
    estimates = np.array([res[1] for res in results], dtype=float)
    stds = np.array([res[2] for res in results], dtype=float)

    ci = [
        confidence_interval(est, std, n, alpha)
        for n, est, std in zip(N, estimates, stds)
    ]

    yerr = np.array([
        [est - lower for est, (lower, _) in zip(estimates, ci)],
        [upper - est for est, (_, upper) in zip(estimates, ci)]
    ])

    plt.figure(figsize=(8.5, 4.8))

    plt.errorbar(
        N,
        estimates,
        yerr=yerr,
        fmt='o-',
        markersize=5,
        linewidth=1.2,
        capsize=4,
        capthick=1.0,
        elinewidth=1.0,
        label="CMC Estimate"
    )

    if reference is not None:
        plt.axhline(
            reference,
            color='red',
            linestyle="--",
            linewidth=1.5,
            label=reference_label
        )

    plt.xscale("log")
    plt.xlabel("Number of samples (N)")
    plt.ylabel(ylabel)
    plt.title(None)
    plt.grid(True, which="both", linestyle=":", linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.show()