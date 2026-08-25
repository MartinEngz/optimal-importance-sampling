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

def plot_vrf_vs_pde_refinement(M_list, P_list, full_results):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for m_val in M_list:
        y_values = [full_results[(p_val, m_val)]['vrf'] for p_val in P_list]
        plt.plot(P_list, y_values, '-o', label=f'M={m_val}')
    plt.xlabel('Space Discretization (P)')
    plt.ylabel('Variance Reduction Factor (VRF)')
    plt.title('Impact of P on VRF')
    plt.legend()
    plt.grid(True, which='both', linestyle=':', linewidth=0.8)

    plt.subplot(1, 2, 2)
    for p_val in P_list:
        y_values = [full_results[(p_val, m_val)]['vrf'] for m_val in M_list]
        plt.plot(M_list, y_values, '-o', label=f'P={p_val}')
    plt.xlabel('Time Discretization (M)')
    plt.ylabel('Variance Reduction Factor (VRF)')
    plt.title('Impact of M on VRF')
    plt.legend()
    plt.grid(True, which='both', linestyle=':', linewidth=0.8)

    plt.tight_layout()
    plt.show()

def plot_barrier_drift_tradeoff(
    r_values,
    estimates,
    stds,
    strike_hits,
    barrier_hits,
    N,
):
    r_labels = [f"{r:.1f}" for r in r_values]
    x = np.arange(len(r_values))
    width = 0.35

    errors = [
        1.96 * std / np.sqrt(N)
        for std in stds
    ]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(
        x - width / 2,
        strike_hits,
        width,
        label="Over strike",
        alpha=0.7,
    )

    ax1.bar(
        x + width / 2,
        barrier_hits,
        width,
        label="Barrier hits",
        alpha=0.7,
    )

    ax1.set_xlabel(r"Fixed drift $\tilde r$")
    ax1.set_ylabel(f"Number of paths (N = {N:,})")
    ax1.set_title(
        "Fixed-Drift Importance Sampling for Up-and-Out Call"
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(r_labels)
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", linestyle=":", linewidth=0.8)

    ax2 = ax1.twinx()

    ax2.errorbar(
        x,
        estimates,
        yerr=errors,
        fmt="o-",
        markersize=5,
        linewidth=1.2,
        capsize=4,
        capthick=1.0,
        elinewidth=1.0,
        label="Price estimate (95% CI)",
    )

    ax2.set_ylabel("Estimated option price")

    fig.tight_layout()
    plt.show()

def plot_barrier_optimal_control(
        t_grid,
        s_grid,
        zeta,
        K,
        U,
        s_min_plot=60.0,
        clip=40.0
):
    t_mask = t_grid <= t_grid[-1]
    s_mask = (
        (s_grid >= s_min_plot)
        & (s_grid <= U)
    )
    
    t_plot = t_grid[t_mask]
    s_plot = s_grid[s_mask]

    zeta_plot = zeta[np.ix_(
        t_mask,
        s_mask,
    )]

    # Zeta* can become very large near the barrier/control singular regions.
    # Therefore, clip to make visualization more readable.
    zeta_plot = np.clip(
        zeta_plot,
        -clip,
        clip
    )

    plt.figure(figsize=(10, 4.8))

    plt.imshow(
        zeta_plot,
        aspect="auto",
        origin="lower",
        extent=[
            s_plot[0],
            s_plot[-1],
            t_plot[0],
            t_plot[-1],
        ]
    )

    plt.colorbar(
        label=r"$\zeta^*(t,s)$"
    )

    plt.axvline(
        K,
        linestyle="--",
        linewidth=1.0,
        label="Strike",
    )

    plt.xlabel("Asset price $s$")
    plt.ylabel("Time $t$")
    plt.title(rf"Optimal control for up-and-out call ($U={U:g}$)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_controlled_barrier_paths(
        paths,
        T,
        K=None,
        U=None,
):
    M = paths.shape[-1] - 1
    t = np.linspace(0.0, T, M + 1)

    plt.figure(figsize=(10, 4.8))

    for path in paths:
        plt.plot(
            t,
            path,
            linewidth=0.7
        )

    if K is not None:
        plt.axhline(
            K,
            linestyle="--",
            label="Strike"
        )

    if U is not None:
        plt.axhline(
            U,
            linestyle="--",
            label="Barrier"
        )

    plt.xlabel("Time")
    plt.ylabel("Asset price")
    plt.title(
        "Controlled paths under optimal importance sampling"
    )

    plt.grid(
        True,
        linestyle=":",
        linewidth=0.8
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_power_price_paths(
    S_paths,
    T,
    K=None,
    max_paths=100,
    title="Simulated Power Price Paths",
):
    n_paths = min(max_paths, S_paths.shape[0])
    t = np.linspace(0.0, T, S_paths.shape[1])

    plt.figure(figsize=(10, 4.8))

    for path in S_paths[:n_paths]:
        plt.plot(
            t,
            path,
            linewidth=0.8,
            alpha=0.7,
        )

    if K is not None:
        plt.axhline(
            K,
            linestyle="--",
            linewidth=1.2,
            label=f"Strike K = {K:g}",
        )
        plt.legend()

    plt.xlabel("Time (years)")
    plt.ylabel("Power price (EUR/MWh)")
    plt.title(title)
    plt.grid(
        True,
        linestyle=":",
        linewidth=0.8,
    )
    plt.tight_layout()
    plt.show()


def plot_power_cap_fixed_is(results):
    """
    Plot how the fixed IS control theta affects
    payoff frequency and variance reduction.
    """
    theta = np.array([res["theta"] for res in results])
    positive_rate = np.array(
        [res["positive_rate"] for res in results]
    )
    vrf = np.array(
        [res["vrf"] for res in results]
    )

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(
        theta,
        100 * positive_rate,
        "o-",
        linewidth=1.2,
        markersize=5,
        label="Positive payoff paths",
    )

    ax1.set_xlabel(r"IS control $\theta$")
    ax1.set_ylabel("Positive payoff paths (%)")
    ax1.grid(
        True,
        linestyle=":",
        linewidth=0.8,
    )

    ax2 = ax1.twinx()

    ax2.plot(
        theta,
        vrf,
        "s--",
        linewidth=1.2,
        markersize=5,
        label="Variance reduction factor",
    )

    ax2.set_ylabel("Variance reduction factor")

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="best",
    )

    plt.title(
        "Fixed Importance Sampling for Power Cap"
    )

    fig.tight_layout()
    plt.show()


def plot_power_cap_adaptive_objective(
    theta,
    objective,
    theta_opt,
):
    plt.figure(figsize=(8.5, 4.8))

    plt.plot(
        theta,
        objective,
        "-",
        linewidth=1.2,
    )

    plt.axvline(
        theta_opt,
        linestyle="--",
        linewidth=1.2,
        label=rf"$\theta^*={theta_opt:.3f}$",
    )

    plt.xlabel(r"IS control $\theta$")
    plt.ylabel(
        r"Estimated second moment $J(\theta)$"
    )

    plt.title(
        "Adaptive Importance Sampling Objective for Power Cap"
    )

    plt.grid(
        True,
        linestyle=":",
        linewidth=0.8,
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_power_price_paths(
    S_paths,
    T,
    K=None,
    max_paths=100,
    title="Simulated Power Price Paths",
):
    n_paths = min(max_paths, S_paths.shape[0])
    t_grid = np.linspace(0.0, T, S_paths.shape[1])

    plt.figure(figsize=(10, 4.8))

    for path in S_paths[:n_paths]:
        plt.plot(
            t_grid,
            path,
            linewidth=0.8,
            alpha=0.7,
        )

    if K is not None:
        plt.axhline(
            K,
            linestyle="--",
            linewidth=1.2,
            label=f"Strike K = {K:g}",
        )
        plt.legend()

    plt.xlabel("Time (years)")
    plt.ylabel("Power price (EUR/MWh)")
    plt.title(title)

    plt.grid(
        True,
        linestyle=":",
        linewidth=0.8,
    )

    plt.tight_layout()
    plt.show()