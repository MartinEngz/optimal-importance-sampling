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
        seasonal_mean,
        rng=None,
        dW=None,
        theta=0.0,
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
    theta : float, optional
            Constant Brownian importance-sampling control.
            theta=0 gives the original measure.
    rng : np.random.Generator, optional
        Random number generator.
    dW : ndarray, optional, shape (N, M)
        Pre-generated Brownian increments. Supplying these allows
        common random numbers across candidate theta values.
        
    Returns
    -------
    S_paths : ndarray, shape (N, M + 1)
        Simulated spot-price paths.
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = T / M
    sqrt_dt = np.sqrt(dt)

    X_paths = np.empty((N, M + 1))
    X_paths[:, 0] = X0

    S_paths = np.empty_like(X_paths)
    S_paths[:, 0] = np.exp(X0)

    if dW is None:
        dW = rng.normal(
            loc=0.0,
            scale=sqrt_dt,
            size=(N, M),
        )

    for m in range(M):
        t_m = m * dt

        X_curr = X_paths[:, m]

        X_paths[:, m + 1] = (
            X_curr + ( kappa * (seasonal_mean(t_m) - X_curr) + sigma * theta ) * dt + sigma * dW[:, m]
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

def simulate_mean_reverting_jump_price_paths(
    N,
    X0,
    kappa,
    sigma,
    T,
    M,
    seasonal_mean,
    jump_intensity,
    jump_scale,
    jump_reversion,
    theta=0.0,
    dW=None,
    jump_increments=None,
    rng=None,
):
    """
    Simulate a seasonal mean-reverting log-price model with
    temporary positive price spikes.

    Model
    -----
        X_t = Y_t + Z_t
        dY_t = [kappa * (mu(t) - Y_t) + sigma * theta] dt
               + sigma dW_t
        dZ_t = -jump_reversion * Z_t dt + dJ_t
    where J_t is a compound Poisson process with intensity
    jump_intensity and exponentially distributed jump sizes.

    Parameters
    ----------
    N : int
        Number of paths.

    X0 : float
        Initial log spot price.

    kappa : float
        Mean-reversion speed of the diffusion factor.

    sigma : float
        Diffusion volatility.

    T : float
        Time horizon in years.

    M : int
        Number of time steps.

    seasonal_mean : callable
        Deterministic seasonal log-price level mu(t).

    jump_intensity : float
        Expected number of jumps per year.

    jump_scale : float
        Mean size of individual exponential jumps in log-price units.

    jump_reversion : float
        Mean-reversion speed of the spike factor.

    theta : float, optional
        Constant Brownian importance-sampling control.
        theta=0 gives the original measure.

    dW : ndarray, optional, shape (N, M)
        Pre-generated Brownian increments. Supplying these allows
        common random numbers across candidate theta values.

    jump_increments : ndarray, optional, shape (N, M)
        Pre-generated compound-Poisson jump increments. Supplying
        these allows the same jump realizations to be reused across
        candidate theta values.

    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    S_paths : ndarray, shape (N, M + 1)
        Simulated electricity spot-price paths.
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = T / M
    sqrt_dt = np.sqrt(dt)

    # --- Brownian increments

    if dW is None:
        dW = rng.normal(
            loc=0.0,
            scale=sqrt_dt,
            size=(N, M),
        )

    # --- Compound-Poisson jump increments

    if jump_increments is None:
        jump_counts = rng.poisson(
            jump_intensity * dt,
            size=(N, M),
        )

        jump_increments = np.zeros((N, M))

        mask = jump_counts > 0

        # Sum of n Exp(scale) variables is Gamma(n, scale)
        jump_increments[mask] = rng.gamma(
            shape=jump_counts[mask],
            scale=jump_scale,
        )

    # --- State variables

    Y_paths = np.empty((N, M + 1))
    Z_paths = np.zeros((N, M + 1))

    Y_paths[:, 0] = X0

    S_paths = np.empty_like(Y_paths)
    S_paths[:, 0] = np.exp(X0)

    # Exact decay factor for the spike component
    jump_decay = np.exp(
        -jump_reversion * dt
    )
    # ---  Simulation
    for m in range(M):
        t_m = m * dt

        Y_curr = Y_paths[:, m]
        Z_curr = Z_paths[:, m]

        Y_paths[:, m + 1] = (
            Y_curr
            + (
                kappa
                * (seasonal_mean(t_m) - Y_curr)
                + sigma * theta
            ) * dt
            + sigma * dW[:, m]
        )

        Z_paths[:, m + 1] = (
            jump_decay * Z_curr
            + jump_increments[:, m]
        )

        S_paths[:, m + 1] = np.exp(
            Y_paths[:, m + 1] + Z_paths[:, m + 1]
        )

    return S_paths
