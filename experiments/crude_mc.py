import numpy as np

from experiments.config import EUROPEAN_CALL, N_LIST, BARRIER_CALL
from src.monte_carlo import mc_european_call_em, mc_up_and_out_call_em
from src.pricing import black_scholes_call_price
from src.statistics import confidence_interval
from src.plotting import (
    plot_standard_error_convergence,
    plot_estimates_with_ci
)

def run_european_call():
    np.random.seed(41)

    cfg = EUROPEAN_CALL

    bs_price = black_scholes_call_price(
        S0=cfg["S0"],
        K=cfg["K"],
        r=cfg["r"],
        sigma=cfg["sigma"],
        T=cfg["T"]
    )

    results = []

    for N in N_LIST:
        est, std = mc_european_call_em(
            N=N,
            S0=cfg["S0"],
            r=cfg["r"],
            sigma=cfg["sigma"],
            T=cfg["T"],
            M=cfg["M"],
            K=cfg["K"]
        )

        results.append((N, est, std))

    print(f"\nReference Black-Scholes price: {bs_price}")
    print("\nCrude Monte Carlo results:\n")

    for N, est, std in results:
        lower, upper = confidence_interval(est, std, N)

        print(
            f"N = {N:>9d} | "
            f"Estimate = {est:.8f} | "
            f"95% CI = [{lower:.8f}, {upper:.8f}] | "
            f"Std = {std:.8f}"
        )

    plot_standard_error_convergence(
        results,
        title="Crude Monte Carlo: convergence rate of standard error"
    )

    plot_estimates_with_ci(
        results,
        reference=bs_price,
        reference_label="Analytical Black-Scholes",
        title="European Call via Crude Monte Carlo"
    )

    return {
        "results": results,
        "bs_price": bs_price,
        "std_reference": results[-1][2]
    }

def run_barrier_call():
    np.random.seed(42)

    cfg = BARRIER_CALL

    results = []

    for N in N_LIST[0:-1]:
        est, std, paths = mc_up_and_out_call_em(
            N=N, 
            S0=cfg["S0"], 
            K=cfg["K"],
            U=cfg["U"],
            r=cfg["r"],
            sigma=cfg["sigma"],
            T=cfg["T"],
            M=cfg["M"]
        )

        over_strike = np.sum(paths[:, -1] > cfg["K"])
        barrier_hits = np.sum(np.max(paths, axis=1) > cfg["U"])

        results.append((N, est, std, over_strike, barrier_hits))

        lower, upper = confidence_interval(est, std, N)
        
        print(
            f"N = {N:>9d} | "
            f"Estimate = {est:.8f} | "
            f"95% CI = [{lower:.8f}, {upper:.8f}] | "
            f"Std = {std:.8f} | "
            f"Over Strike = {over_strike} | "
            f"Barrier Hits={barrier_hits} | "
        )

    std_reference=results[-1][2]

    return {
        "results": results,
        "std_reference": std_reference
    }
    



if __name__ == "__main__":
    # run_european_call()
    run_barrier_call()