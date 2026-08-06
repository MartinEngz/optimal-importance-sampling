# is_fixed_rtilde

import numpy as np
from scipy.stats import norm
from src.models import log_price_params
from src.payoffs import european_call_payoff

def is_fixed_rtilde(N, S0, r, r_tilde, sigma, T):
    """
    Importance sampling estimator for the European call by changing the drift
    from r to r_tilde. Implementation follows Example 6.4 of lecture notes to
    MATH-414.

    Returns:
        estimate, std_estimator
    """
    mu_r, var = log_price_params(r, sigma, T)
    mu_rt, _ = log_price_params(r_tilde, sigma, T)

    # Sample under the IS distribution: N(mu_rt, var)
    X_tilde = np.random.randn(N) * np.sqrt(var) + mu_rt
    S_T_tilde = S0 * np.exp(X_tilde)

    # Payoffs under original payoff function
    payoffs = european_call_payoff(S_T_tilde)

    # Likelihood ratio w(x) = f_r(x) / f_rtilde(x)
    std = np.sqrt(var)
    f_r_vals = norm.pdf(X_tilde, loc=mu_r, scale=std)
    f_rt_vals = norm.pdf(X_tilde, loc=mu_rt, scale=std)
    w = f_r_vals / f_rt_vals

    weighted_payoffs = payoffs * w
    est = np.mean(weighted_payoffs)
    est_std = np.std(weighted_payoffs, ddof=1)
    return est, std
