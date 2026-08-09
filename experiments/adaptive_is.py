import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from experiments.config import EUROPEAN_CALL, ADAPTIVE_IS
from src.importance_sampling import adaptive_is_rtilde
from src.models import log_price_params
from src.pricing import black_scholes_call_price
from src.statistics import variance_reduction, absolute_error

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

if __name__ == "__main__":
    run_european_call()
