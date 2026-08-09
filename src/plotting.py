import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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


def plot_optimal_control(t_grid, s_grid, zeta_star, t_max=0.18, s_range=None):
    """
    Plots the optimal control zeta* using Plotly.
    
    Parameters:
    t_grid: 1D array of time points from 0 to T.
    s_grid: 1D array of asset price points from S_min to S_max.
    zeta_star: 2D array of control values (M+1, P+1).
    t_max: Upper bound for time axis (default 0.18).
    s_range: Tuple (S1, S2) for price axis filtering.
    """

    t_indices = np.where(t_grid <= t_max)[0]
    
    if s_range is not None:
        s1, s2 = s_range
        s_indices = np.where((s_grid >= s1) & (s_grid <= s2))[0]
    else:
        s_indices = np.arange(len(s_grid))

    t_plot = t_grid[t_indices]
    s_plot = s_grid[s_indices]
    
    # Z_plot: Rows correspond to Asset Price (y), Columns to Time (x)
    z_plot = zeta_star[np.ix_(t_indices, s_indices)].T

    fig = go.Figure(data=[go.Surface(
        z=z_plot, 
        x=t_plot, 
        y=s_plot,
        colorscale='Viridis',
        colorbar_title='ζ*'
    )])

    fig.update_layout(
        title=f'Optimal Control ζ*(t, s) for t ∈ [0, {t_max}]',
        scene=dict(
            xaxis_title='Time (t)',
            yaxis_title='Asset Price (s)',
            zaxis_title='Control ζ*',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.7)
        ),
        width=900,
        height=700,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    fig.show()

