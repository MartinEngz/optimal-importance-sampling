# crude MC estimators

import numpy as np

from src.models import (
    simulate_gbm_em_paths,
    simulate_mean_reverting_log_price_paths,
)

from src.payoffs import (
    european_call_payoff, 
    up_and_out_call_payoff,
    power_cap_payoff,
)

def mc_european_call_em(N, S0, r, sigma, T, M, K):
    """
    Crude MC estimator of vanilla European call using Euler-Maruyama.

    Returns:
        estimate, std_estimator
    """
    S_paths, _ = simulate_gbm_em_paths(N, S0, r, sigma, T, M)
    S_T = S_paths[:, -1]
    payoffs = european_call_payoff(S_T, K, r, T)
    est = np.mean(payoffs)
    std = np.std(payoffs, ddof=1)
    return est, std

def mc_up_and_out_call_em(N, S0, K, U, r, sigma, T, M):
    """
    Crude MC estimator of up-and-out European call using Euler-Maruyama.

    Returns:
        estimate, std_estimator 
    """
    S_paths, _ = simulate_gbm_em_paths(N, S0, r, sigma, T, M)

    payoffs = up_and_out_call_payoff(S_paths, K, r, T, U)

    estimate = np.mean(payoffs)
    std = np.std(payoffs, ddof=1)

    return estimate, std, S_paths

def mc_power_cap(
    N,
    X0,
    K,
    kappa,
    sigma,
    T,
    M,
    r,
    volume,
    delivery_indices,
    seasonal_mean,
):
    S_paths = simulate_mean_reverting_log_price_paths(
        N=N,
        X0=X0,
        kappa=kappa,
        sigma=sigma,
        T=T,
        M=M,
        seasonal_mean=seasonal_mean,
    )

    payoffs = power_cap_payoff(
        S_paths=S_paths,
        K=K,
        delivery_indices=delivery_indices,
        r=r,
        T=T,
        M=M,
        volume=volume,
    )

    estimate = np.mean(payoffs)
    std = np.std(payoffs, ddof=1)

    return estimate, std, S_paths
