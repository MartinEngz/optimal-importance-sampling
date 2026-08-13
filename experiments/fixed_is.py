import numpy as np

from experiments.config import EUROPEAN_CALL, N_LIST
from src.importance_sampling import is_fixed_rtilde
from src.pricing import black_scholes_call_price
from src.statistics import (
    confidence_interval,
    variance_reduction,
    absolute_error
)
from src.plotting import plot_estimates_with_ci

def run_european_call(std_reference=None):
    cfg=EUROPEAN_CALL

    bs_price = black_scholes_call_price(
        S0=cfg["S0"],
        K=cfg["K"],
        r=cfg["r"],
        sigma=cfg["sigma"],
        T=cfg["T"]
    )
    

    # We choose the IS drift such that log-price is centered at log(K)
    r_tilde = (
        np.log(cfg["K"] / cfg["S0"] / cfg["T"] + 0.5 * cfg["sigma"] ** 2)
    )

    print(f"\nUsing r_tilde = {r_tilde:.4f}")

    results = []

    for N in N_LIST:
        est, std = is_fixed_rtilde(
            N,
            cfg["S0"],
            cfg["K"],
            cfg["r"],
            r_tilde,
            cfg["sigma"],
            cfg["T"]
        )

        results.append((N, est, std))

    print("\nFixed Importance Sampling results:\n")

    for N, est, std in results:
        lower, upper = confidence_interval(est, std, N)

        print(
            f"N = {N:>9d} | "
            f"Estimate = {est:.8f} | "
            f"95% CI = [{lower:.8f}, {upper:.8f}] | "
            f"Std = {std:.8f} | "
            f"Abs error = {absolute_error(est, bs_price):.8f}"
        )

    if std_reference is not None:
        vrf = variance_reduction(
            std_reference, 
            results[-1][2]
        )

        print(f"\nVariance reduction vs CMC: {vrf:.2f}")

    else:
        vrf = None

    plot_estimates_with_ci(
        results,
        reference=bs_price,
        reference_label="Analytical Black-Scholes",
        title="European Call via Fixed Importance Sampling"
    )

    return {
        "results": results,
        "r_tilde": r_tilde,
        "variance_reduction": vrf
    }

def run_barrier_call(std_reference=None):
    pass

    # std_err = standard_error(std, N)


if __name__ == "__main__":
    run_european_call()