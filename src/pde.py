# finite-difference Kolmogorov backward equation solver
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def solve_european_call_kbe(S0, K, T, r, sigma, M, P):
    """
    Solve the Black-Scholes Kolmogorov backward equation
    for the discounted European call payoff.

    Returns:
        s_grid, t_grid, v
    """
    # PDE domain
    S_min = S0 * np.exp((r - 0.5 * sigma**2) * T - 6 * sigma * np.sqrt(T))
    S_max = S0 * np.exp((r - 0.5 * sigma**2) * T + 6 * sigma * np.sqrt(T))

    # Boundary conditions and discretization initialization
    s_grid = np.linspace(S_min, S_max, P + 1)
    t_grid = np.linspace(0, T, M + 1)
    ds = (S_max - S_min) / P
    dt = T / M


    v = np.zeros((M + 1, P + 1))
    v[M, :] = np.maximum(s_grid - K, 0) * np.exp(-r*T) # Terminal condition v(T,s) = psi(s)

    # Matrix assembly for implicit euler
    s_inner = s_grid[1:-1] 
    L = (dt * r * s_inner / (2 * ds)) - (dt * sigma**2 * s_inner**2 / (2 * ds**2))
    D = 1 + (dt * sigma**2 * s_inner**2 / ds**2)
    U = ( -dt * r * s_inner / (2 * ds) - dt * sigma**2 * s_inner**2 / (2 * ds**2) )
    A = diags([L[1:], D, U[:-1]], [-1, 0, 1], shape=(P-1, P-1), format='csc')

    # Backward time-stepping
    for m in range(M - 1, -1, -1):
        t_m = t_grid[m]
        rhs = v[m + 1, 1:-1].copy()

        # BCs
        v_min = 0.0
        v_max = S_max - K * np.exp(-r * (T - t_m))

        rhs[0] -= L[0] * v_min
        rhs[-1] -= U[-1] * v_max

        v[m, 1:-1] = spsolve(A, rhs)
        v[m, 0] = v_min
        v[m, -1] = v_max

    return s_grid, t_grid, v

def solve_barrier_call_kbe(
    S0,
    K,
    U,
    T,
    r,
    sigma,
    M,
    P,
    epsilon_bc=0.1,
    smin_sigma_mult=6.0,
):
    """
    Solves the KBE for an up-and-out European call.

    Domain:
        s in [S_min, U]

    Boundary conditions:
        v(t, S_min) = 0
        v(t, U) = epsilon_bc
    Terminal condition:
        v(T, s) = exp(-rT) * max(s - k, 0)
    """
    S_min = S0 * np.exp(
        (r - 0.5 * sigma**2) * T
        - smin_sigma_mult * sigma_np.sqrt(T)
    )
    S_max = U

    ds = (S_max - S_min) / P
    dt = T / M

    s_grid = np.linspace(S_min, S_max, P + 1)
    t_grid = np.linspace(0.0, T, M + 1)

    v = np.zeros((M + 1, P + 1))

    # Terminal conditions
    v[:, 0] = 0.0
    v[:, -1] = epsilon_bc

    s_inner = s_grid[1:-1]

    lower = (
        dt * r * s_inner / (2 * ds)
        - dt * sigma**2 * s_inner**2
    )

    diagonal = (
        1.0 + dt * sigma**2 * s_inner**2 / (2 * ds**2)
    )

    upper = (
        -dt * r * s_inner / (2 * ds)
        - dt * sigma**2 * s_inner**2 / (2 * ds**2)
    )

    A = diags(
        [
            lower[1:],
            diagonal,
            upper[:-1]
        ],
        offset=[-1, 0, 1],
        format="csc",
    )

    for m in range(M - 1, -1, -1):
        rhs = v[m + 1, 1:-1].copy()

        # Boundary contributions
        # rhs[0] -= lower[0] * 0.0          # <-- gives -= 0.0
        rhs[-1] -= upper[-1]
        v[m, 1:-1] = spsolve(A, rhs)
        v[m, 0] = 0.0
        v[m, -1] = epsilon_bc

    return s_grid, t_grid, v

