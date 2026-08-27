import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from experiments.config import (
    EUROPEAN_CALL, 
    ADAPTIVE_IS,
    POWER_CAP,
    POWER_CAP_ADAPTIVE_IS,
    POWER_CAP_JUMP
)
from src.importance_sampling import (
    adaptive_is_rtilde,
    adaptive_is_theta_power_cap
)
from src.models import (
    log_price_params,
    sinusoidal_log_seasonality,
)

from src.pricing import black_scholes_call_price

from src.statistics import (
    variance_reduction, 
    absolute_error,
)

from src.plotting import plot_power_cap_adaptive_objective

def plot_objective_sanity_check(result):
    """
    Algorithm specific objective diagnostics.
    """
    X = result["last_X"]

    if X is None:
        print("No sample stored for J(theta) sanity check.")
        return

    payoffs = result["last_payoffs"]
    f_r_vals = result["last_f_r_vals"]
    f_rt_vals = result["last_f_rt_vals"]

    r_final = result["r_tilde_final"]
    r_min = result["r_min"]
    r_max = result["r_max"]
    hw = result["refine_half_width"]

    var_X = result["var_X"]
    std_X = np.sqrt(var_X)

    lo = max(r_min, r_final - hw)
    hi = min(r_max, r_final + hw)

    grid = np.linspace(lo, hi, result["n_refine"])
    J = np.empty_like(grid)

    f_r_sq = f_r_vals ** 2

    for j, r_candidate in enumerate(grid):
        mu_cand, _ = log_price_params(
            r_candidate,
            result["sigma"],
            result["T"]
        )

        f_cand = norm.pdf(
            X,
            loc=mu_cand,
            scale=std_X
        )

        denom = ( f_cand * f_rt_vals + result["eps_denom"])

        J[j] = np.mean((payoffs ** 2) * f_r_sq / denom)

    plt.figure(figsize=(9, 4.8))
    plt.plot(grid, J)
    plt.axvline(
        r_final,
        linestyle="--",
        label=r"final $\tilde r$"
    )
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\hat J(\theta)$")
    plt.title(r"Sanity check: Adaptive IS objective $\hat J(\theta)$ around the minimizer")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_european_call(std_reference=None):
    cfg=EUROPEAN_CALL
    adaptive_cfg=ADAPTIVE_IS

    result = adaptive_is_rtilde(
        S0=cfg["S0"],
        K=cfg["K"],
        r=cfg["r"],
        sigma=cfg["sigma"],
        T=cfg["T"],
        **adaptive_cfg
    )

    bs_price = black_scholes_call_price(
        cfg["S0"],
        cfg["K"],
        cfg["r"],
        cfg["sigma"],
        cfg["T"]
    )

    print("\nFinal adaptive results:")
    print("r_tilde_final =", result["r_tilde_final"])
    print("N_final       =", result["N_final"])
    print("Z_hat         =", result["Z_hat"])
    print("sigma_hat     =", result["sigma_hat"])
    print(
        "Absolute error =",
        absolute_error(result["Z_hat"], bs_price)
    )

    if std_reference is not None:
        vrf = variance_reduction(
            std_reference,
            result["sigma_hat"]
        )

        print(f"Variance reduction vs CMC: {vrf:.2f}")

    else:
        vrf = None

    plot_objective_sanity_check(result)

    return {
        "result": result,
        "variance_reduction": vrf
    }


def run_power_cap(std_reference=None):
    cfg = POWER_CAP
    adaptive_cfg = POWER_CAP_ADAPTIVE_IS
    jump_model_cfg = POWER_CAP_JUMP

    X0 = np.log(cfg["S0"])

    seasonal_mean = lambda t: sinusoidal_log_seasonality(
        t=t,
        base_price=cfg["base_price"],
        amplitude=cfg["seasonal_amplitude"],
        phase=cfg["seasonal_phase"],
    )

    delivery_indices = np.arange(1, cfg["M"] + 1)

    result = adaptive_is_theta_power_cap(
        X0=X0,
        K=cfg["K"],
        kappa=cfg["kappa"],
        sigma=cfg["sigma"],
        T=cfg["T"],
        M=cfg["M"],
        r=cfg["r"],
        volume=cfg["volume"],
        delivery_indices=delivery_indices,
        seasonal_mean=seasonal_mean,
        **adaptive_cfg,
        **jump_model_cfg
    )

    print("\n--- Adaptive IS: Power Cap ---")

    print(
        f"Optimal theta:        "
        f"{result['theta']:.4f}"
    )

    print(
        "Price estimate:        "
        f"{result['estimate']:.8e}"
    )

    print(
        f"Weighted payoff std:  "
        f"{result['std']:.8e}"
    )

    print(
        f"Positive paths:       "
        f"{result['positive_paths']:,}"
        f"({result['positive_rate']:.2%})"
    )

    vrf = None

    if std_reference is not None:
        vrf = variance_reduction(
            std_reference=std_reference,
            std_method=result["std"]
        )

        print(
            f"Variance reduction: "
            f"{vrf:.2f}"
        )

    result["variance_reduction"] = vrf

    plot_power_cap_adaptive_objective(
        theta=result["theta_grid"],
        objective=result["objective_grid"],
        theta_opt=result["theta"],
    )

    return result

if __name__ == "__main__":
    # run_european_call()
    # run_power_cap(std_reference=0.001252) #OBS: mc std reference without jumps: 0.001252

    
    # Results for adaptive IS on mean-reverting only model

    # Iteration  1 | N=  1,000 | theta_opt=11.8000 | J(theta)=1.212915e-09
    # Iteration  2 | N=  2,000 | theta_opt=11.6750 | J(theta)=1.171739e-09
    # Iteration  3 | N=  4,000 | theta_opt=15.2250 | J(theta)=1.661158e-09
    # Iteration  4 | N=  8,000 | theta_opt=14.9000 | J(theta)=1.550510e-09
    # Iteration  5 | N= 16,000 | theta_opt=11.3500 | J(theta)=2.001198e-09
    # Iteration  6 | N= 32,000 | theta_opt=11.7500 | J(theta)=1.768160e-09
    # Iteration  7 | N= 64,000 | theta_opt=11.6000 | J(theta)=1.597664e-09
    # Iteration  8 | N=128,000 | theta_opt=13.1750 | J(theta)=2.009877e-09
    # Iteration  9 | N=256,000 | theta_opt=11.0500 | J(theta)=1.947082e-09
    # Iteration 10 | N=512,000 | theta_opt=10.9750 | J(theta)=1.987006e-09

    # --- Adaptive IS: Power Cap ---
    # Optimal theta:        10.9750
    # Price estimate:        1.53852528e-05
    # Weighted payoff std:  4.08309528e-05
    # Positive paths:       417,631(40.78%)
    # Variance reduction: 940.22  

    run_power_cap(std_reference=0.138714) # OBS: mc std reference with jumps: 0.138714
    
    