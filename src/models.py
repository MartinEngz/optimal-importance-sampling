# GBM / SDE simulation

import numpy as np

def simulate_gbm_em_paths(N, S0, r, sigma, T, M):
    """
    Simulate N paths of GBM using Euler-Maruyama.

    Returns:
        S_paths: array of shape (N, M+1) with full paths.
        W_T: array of shape(N, 1) with final W value.
    """
    dt = T / M
    sqrt_dt = np.sqrt(dt)

    S_paths = np.empty((N, M+1))
    S_paths[:, 0] = S0

    W_T = np.zeros(N)
    for m in range(M):
        Z = np.random.randn(N)
        dW = sqrt_dt * Z
        S = S_paths[:, m]
        S_paths[:, m+1] = S + r * S * dt + sigma * S * dW
        W_T += dW
    return S_paths, W_T

def log_price_params(r_param, sigma, T):
    """
    Mean and variance of log(S_T / S0) under Black-Scholes with rate r_param.
    """
    mu = (r_param - 0.5 * sigma ** 2) * T
    var = sigma ** 2 * T
    return mu, var