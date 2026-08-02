# GBM / SDE simulation

import numpy as np

def simulate_gbm_em_paths(N, S0, r, sigma, T, M):
    """
    Simulate N paths of GBM using Euler-Maruyama.

    Returns:
        S_paths: array of shape (N, M+1) with full paths.
    """
    dt = T / M
    sqrt_dt = np.sqrt(dt)

    S_paths = np.empty((N, M+1))
    S_paths[:, 0] = S0

    for m in range(M):
        Z = np.random.randn(N)
        dW = sqrt_dt * Z
        S = S_paths[:, m]
        S_paths[:, m+1] = S + r * S * dt + sigma * S * dW
    return S_paths
