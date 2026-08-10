# crude MC estimators

import numpy as np

from src.models import simulate_gbm_em_paths
from src.payoffs import european_call_payoff, up_and_out_call_payoff

def mc_european_call_em(N, S0, r, sigma, T, M, K):
    """
    Crude MC estimator of vanilla European call using Euler-Maruyama.

    Returns:
        estimate, std_estimator
    """
    S_paths = simulate_gbm_em_paths(N, S0, r, sigma, T, M)
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
    S_paths = simulate_gbm_em_paths(N, S0, r, sigma, T, M)

    payoffs = up_and_out_call_payoff(S_paths, K, r, T, U)

    estimate = np.mean(payoffs)
    std = np.std(payoffs, ddof=1)

    return estimate, std, S_paths