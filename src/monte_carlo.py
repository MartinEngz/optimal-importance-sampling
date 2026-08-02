# crude MC estimators

import numpy as np

from src.models import simulate_gbm_em_paths
from src.payoffs import european_call_payoff

def mc_european_call_em(N, S0, r, sigma, T, M, K):
    """
    Crude MC estimator using Euler-Maruyama.

    Returns:
        estimate, std_estimator
    """
    S_paths = simulate_gbm_em_paths(N, S0, r, sigma, T, M)
    S_T = S_paths[:, -1]
    payoffs = european_call_payoff(S_T, K, r, T)
    est = np.mean(payoffs)
    payoff_std = np.std(payoffs, ddof=1)
    return est, payoff_std