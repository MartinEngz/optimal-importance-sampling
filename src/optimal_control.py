# controlled SDE + Girsanov weights

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.statistics import variance_reduction

def compute_optimal_control(v, s_grid, sigma):
    """
    Compute zeta* = sigma * s * ∂s log(v)
    """
    ds = s_grid[1] - s_grid[0]

    dv_ds = np.zeros_like(v)

    dv_ds[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * ds)

    dv_ds[:, 0] = ( v[:, 1] - v[:, 0]) / ds
    dv_ds[:, -1] = ( v[:, -1] - v[:, -2]) / ds

    safe_v = np.maximum(v, 1e-14)

    zeta = (sigma * s_grid[None, :] * dv_ds / safe_v)

    # Prolongate control from S_min + ds to avoid divergence at zero boundary
    zeta[:, 0] = zeta[:, 1]
    # Set control to zero at final time T
    zeta[-1, :] = 0.0

    return zeta

def get_control_interpolator(t_grid, s_grid, zeta):
    """
    Creates a piecewise linear interpolator for the optimal control zeta*.

    Parameters:
    t_grid: 1D array of time points from 0 to T.
    s_grid: 1D array of asset price points from S_min to S_max.
    zeta_star: 2D array (matrix) of control values computed on the grid.
    """
    interp = RegularGridInterpolator((t_grid, s_grid), zeta, method='linear', bounds_error=False, fill_value=None)

    t_min, t_max = t_grid[0], t_grid[-1]
    s_min, s_max = s_grid[0], s_grid[-1]

    def control(t, s):
        t_clipped = np.clip(t, t_min, t_max)
        s_clipped = np.clip(s, s_min, s_max)

        return interp((t_clipped, s_clipped))

    return control

def simulate_controlled_european_call(
        N, 
        S0, 
        K, 
        T, 
        r, 
        sigma, 
        M, 
        control_func
    ):
    """
    Vectorized Importance Sampling for European Call Option using Optimal Control.
    """
    dt = T / M
    sqrt_dt = np.sqrt(dt)
    
    S_curr = np.full(N, S0)
    log_L = np.zeros(N)
    
    for m in range(M):
        t_m = m * dt
        
        zeta_m = control_func(t_m, S_curr)
        dW = np.random.normal(0, sqrt_dt, size=N)
        log_L += -0.5 * (zeta_m**2) * dt - zeta_m * dW
        
        S_curr += (r * S_curr + sigma * S_curr * zeta_m) * dt + sigma * S_curr * dW
        
    payoffs = np.maximum(S_curr - K, 0) * np.exp(-r * T)
    weighted_payoffs = payoffs * np.exp(log_L)
    
    Z_is = np.mean(weighted_payoffs)
    Var_is = np.var(weighted_payoffs, ddof=1)
    error_is = 1.96 * np.sqrt(Var_is / N)
    
    return Z_is, Var_is, error_is