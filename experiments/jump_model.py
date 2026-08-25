import numpy as np

from experiments.config import (
    POWER_CAP,
    POWER_CAP_JUMP,
)

from src.models import (
    sinusoidal_log_seasonality,
    simulate_mean_reverting_jump_price_paths,
)

from src.plotting import (
    plot_power_price_paths,
)


def run_jump_path_simulation(
    N=100,
    random_seed=123,
):
    """
    Simulate and visualize power-price paths under the
    mean-reverting diffusion + temporary spike model.
    """
    cfg = POWER_CAP
    jump_cfg = POWER_CAP_JUMP

    rng = np.random.default_rng(random_seed)

    X0 = np.log(cfg["S0"])

    seasonal_mean = lambda t: sinusoidal_log_seasonality(
        t=t,
        base_price=cfg["base_price"],
        amplitude=cfg["seasonal_amplitude"],
        phase=cfg["seasonal_phase"],
    )

    S_paths = simulate_mean_reverting_jump_price_paths(
        N=N,
        X0=X0,
        kappa=cfg["kappa"],
        sigma=cfg["sigma"],
        T=cfg["T"],
        M=cfg["M"],
        seasonal_mean=seasonal_mean,
        jump_intensity=jump_cfg["jump_intensity"],
        jump_scale=jump_cfg["jump_scale"],
        jump_reversion=jump_cfg["jump_reversion"],
        theta=0.0,       # original measure
        rng=rng,
    )

    plot_power_price_paths(
        S_paths=S_paths,
        T=cfg["T"],
        K=cfg["K"],
        max_paths=N,
        title="Mean-Reverting Power Price Paths with Spikes",
    )

    # Simple diagnostics
    max_prices = np.max(S_paths, axis=1)

    print("\n--- Jump-model path simulation ---")
    print(f"Number of paths:        {N:,}")
    print(f"Initial price:          {cfg['S0']:.2f} EUR/MWh")
    print(f"Strike:                 {cfg['K']:.2f} EUR/MWh")
    print(
        f"Paths exceeding strike: "
        f"{np.sum(max_prices > cfg['K']):,} "
        f"({np.mean(max_prices > cfg['K']):.2%})"
    )
    print(
        f"Maximum simulated price: "
        f"{np.max(S_paths):.2f} EUR/MWh"
    )

    return {
        "paths": S_paths,
        "max_prices": max_prices,
    }


if __name__ == "__main__":
    run_jump_path_simulation()