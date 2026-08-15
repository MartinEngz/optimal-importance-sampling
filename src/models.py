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

def simulate_mean_reverting_log_price_paths(
        N,
        X0,
        kappa,
        sigma,
        T,
        M,
        seasonal_mean
):
    """
    Simulate a mean-reverting log-price process

        dX_t = kappa * (mu(t) - X_t) dt + sigma dW_t

    using Euler-Maruyama, with S_t = exp(X_t).

    Parameters
    ----------
    N: int
        Number of simulated paths.
    X0 : float
        Initial log spot price.
    kappa: float
        Mean-reversion speed.
    sigma : float
        Time horizon in years.
    M : int
        Number of time steps.
    seasonal_mean : callable
        Function mu(t) returning the seasonal log-price level.

    Returns
    -------
    S_paths : ndarray, shape (N, M + 1)
        Simulated spot-price pahts.
    """
    dt = T / M
    sqrt_dt = np.sqrt(dt)

    X_paths = np.empty((N, M + 1))
    X_paths[:, 0] = X0

    S_paths = np.empty_like(X_paths)
    S_paths[:, 0] = np.exp(X0)

    for m in range(M):
        t_m = m * dt

        dW = np.random.normal(
            0.0,
            sqrt_dt,
            size=N,
        )

        X_curr = X_paths[:, m]

        X_paths[:, m + 1] = (
            X_curr + kappa * (seasonal_mean(t_m) - X_curr) * dt + sigma * dW
        )
        S_paths[:, m + 1] = np.exp(X_paths[:, m + 1])

    return S_paths

def sinusoidal_log_seasonality(
        t,
        base_price,
        amplitude,
        phase=0.0,
):
    """
    Deterministic seasonal mean for log electricity prices.
    """
    return (
        np.log(base_price) + amplitude * np.cos(2 * np.pi * (t - phase))
    )


