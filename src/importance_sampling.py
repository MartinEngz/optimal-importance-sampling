import numpy as np

from scipy.stats import norm

from src.models import log_price_params

from src.payoffs import (
    european_call_payoff, 
    up_and_out_call_payoff,
    power_cap_payoff,
)

from src.models import (
    simulate_gbm_em_paths,
    simulate_mean_reverting_log_price_paths,
)

from src.statistics import standard_error

def is_fixed_rtilde(N, S0, K, r, r_tilde, sigma, T):
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
    payoffs = european_call_payoff(S_T_tilde, K, r, T)

    # Likelihood ratio w(x) = f_r(x) / f_rtilde(x)
    std = np.sqrt(var)
    f_r_vals = norm.pdf(X_tilde, loc=mu_r, scale=std)
    f_rt_vals = norm.pdf(X_tilde, loc=mu_rt, scale=std)
    w = f_r_vals / f_rt_vals

    weighted_payoffs = payoffs * w
    est = np.mean(weighted_payoffs)
    std = np.std(weighted_payoffs, ddof=1)
    return est, std


def is_fixed_rtilde_barrier(N, S0, K, U, r, r_tilde, sigma, T, M, random_seed=None):
    """
    Importance sampling estimator for the Barrier call by changing the drift from
    r to r_tilde.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    S_paths, W_T = simulate_gbm_em_paths(N, S0, r_tilde, sigma, T, M)
    payoffs = up_and_out_call_payoff(S_paths, K, r, T, U)

    theta = (r - r_tilde) / sigma
    weights = np.exp(-theta * W_T - 0.5 * (theta**2) * T)

    weighted_payoffs = payoffs * weights
    est = np.mean(weighted_payoffs)
    std = np.std(weighted_payoffs, ddof=1)

    return est, std, S_paths


def adaptive_is_rtilde(
        S0, K, r, sigma, T,
        tol=1e-3, alpha=0.05,
        N_bar=1000, gamma=2.0,
        r_min=0.05, r_max=20.0,
        n_grid=100,                 # coarse grid size
        n_refine=501,               # refined grid size around coarse minimizer
        refine_half_width=0.25,     # refine interval: [best-hw, best+hw]
        max_iter=20,
        random_seed=123,
        eps_denom=1e-300
):
    """
    Adaptive IS for European call with variance-minimization objective J(theta)
    following Algorithm 6.3/6.4 from the lecture notes of MATH-414.

    The minimization is performed via a two-stage grid search, first minimizing
    the empirical J(theta) over a course grid, then refining around the best
    candidate.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    c = norm.ppf(1 - alpha / 2.0)

    # Precompute mean and variance of X = log(S_T/S0) under the true model r
    mu_r, var_X = log_price_params(r, sigma, T)

    ## Step 1 of Algorithm 6.4
    # Initialize
    N = int(N_bar / gamma)
    r_tilde = r
    sigma_hat = np.inf

    # Histories
    r_tilde_history = []
    N_history = []
    Z_hat_history = []
    sigma_hat_history = []

    # Keep last sample to allow an external J(theta) sanity-check plot
    last_X = None
    last_payoffs = None
    last_f_r_vals = None
    last_f_rt_vals = None # f(x; current r_tilde) at time of sampling

    for it in range(max_iter):
        halfwidth = c * sigma_hat / np.sqrt(N)
        print(f"Iteration {it+1}: N={N}, r_tilde={r_tilde:.4f}, sigma_hat={sigma_hat:.6f}")
        print(f"  Current tolerance check (half-width vs tol): {halfwidth:.6f} <= {tol:.6f}\n")

        ## Step 2 of Algorithm 6.4
        if halfwidth <= tol:
            print("** Desired tolerance achieved! **")
            break

        ## Step 3 of Algorithm 6.4
        # Update r_tilde and N (r_tilde update later in order not to print previous value)
        N = int(gamma * N)

        ## Step 4 of Algorithm 6.4
        # Sample under current proposal f(·; r_tilde)
        mu_rt, _ = log_price_params(r_tilde, sigma, T)
        X = np.random.randn(N) * np.sqrt(var_X) + mu_rt
        S_T = S0 * np.exp(X)

        # IS weights w = f_r / f_rtilde
        std_X = np.sqrt(var_X)
        f_r_vals = norm.pdf(X, loc=mu_r, scale=std_X)
        f_rt_vals = norm.pdf(X, loc=mu_rt, scale=std_X)
        w = f_r_vals / (f_rt_vals + eps_denom)

        payoffs = european_call_payoff(S_T, K, r, T)
        weighted_payoffs = payoffs * w

        ## Step 5 of Algorithm 6.4
        # MC IS estimator and variance under current r_tilde
        Z_hat = np.mean(weighted_payoffs)
        sigma_hat = np.std(weighted_payoffs, ddof=1)

        # store last sample info (for external plotting later)
        last_X = X
        last_payoffs = payoffs
        last_f_r_vals = f_r_vals
        last_f_rt_vals = f_rt_vals

        ## Step 6 of Algorithm 6.4
        # Approximate J(theta) and choose new r_tilde minimizing J_hat(theta) on a grid

        # Build coarse grid and compute J_coarse explicitly
        r_grid_coarse = np.linspace(r_min, r_max, n_grid)
        J_coarse = np.empty_like(r_grid_coarse, dtype=float)

        f_r_sq = f_r_vals ** 2 # common factor

        for j, r_candidate in enumerate(r_grid_coarse):
            mu_cand, _ = log_price_params(r_candidate, sigma, T)
            f_cand = norm.pdf(X, loc=mu_cand, scale=std_X)
            denom = (f_cand * f_rt_vals) + eps_denom
            J_coarse[j] = np.mean((payoffs ** 2) * f_r_sq / denom)

        # Print top-5 coarse candidates
        idx5 = np.argsort(J_coarse)[:5]
        print("  Top-5 coarse candidates (r, J_hat):")
        for k in idx5:
            print(f"    r={r_grid_coarse[k]:.6f}, J={J_coarse[k]:.6e}")
        best_coarse = float(r_grid_coarse[np.argmin(J_coarse)])
        print(f"  Coarse minimizer: r ~{best_coarse:.6f}\n")

        # Refine grid around coarse minimizer
        lo = max(r_min, best_coarse - refine_half_width)
        hi = min(r_max, best_coarse + refine_half_width)
        r_grid_fine = np.linspace(lo, hi, n_refine)
        J_fine = np.empty_like(r_grid_fine, dtype=float)

        for j, r_candidate in enumerate(r_grid_fine):
            mu_cand, _ = log_price_params(r_candidate, sigma, T)
            f_cand = norm.pdf(X, loc=mu_cand, scale=std_X)
            denom = (f_cand * f_rt_vals) + eps_denom
            J_fine[j] = np.mean((payoffs**2) * f_r_sq / denom)

        best_fine = float(r_grid_fine[np.argmin(J_fine)])
        print(f"  Refined minimizer in [{lo:.4f}, {hi:.4f}]: r_tilde_new = {best_fine:.6f}\n")

        # Update r_tilde
        r_tilde = best_fine

        # Store UPDATED value
        r_tilde_history.append(r_tilde)
        N_history.append(N)
        Z_hat_history.append(Z_hat)
        sigma_hat_history.append(sigma_hat)

    return {
        "r_tilde_history": r_tilde_history,
        "N_history": N_history,
        "Z_hat_history": Z_hat_history,
        "sigma_hat_history": sigma_hat_history,
        "r_tilde_final": r_tilde,
        "N_final": N,
        "Z_hat": Z_hat_history[-1] if Z_hat_history else None,
        "sigma_hat": sigma_hat_history[-1] if sigma_hat_history else None,
        # extra info for external J(theta) sanity check plot
        "last_X": last_X,
        "last_payoffs": last_payoffs,
        "last_f_r_vals": last_f_r_vals,
        "last_f_rt_vals": last_f_rt_vals,
        "var_X": var_X,
        "mu_r": mu_r,
        "r_min": r_min,
        "r_max": r_max,
        "refine_half_width": refine_half_width,
        "n_refine": n_refine,
        "eps_denom": eps_denom,
        "sigma": sigma,
        "T": T,
    }


def is_fixed_theta_power_cap(
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
        theta,
        random_seed=None
):
    """
    Fixed-control importance sampling for a cap-style power derivative.

    Simulates

        dX_T = [kappa(mu(t) - X_t) + sigma*theta] dt + sigma dW_t
    
    and reweights with the Girsanov likelihood ratio

        L_T = exp(-theta W_T - 0.5 theta^2 T).

    Obs: this is kept self-contained rather than complicating the existing model
    simulator by changing its drift term and exposing W_T.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    dt = T / M
    sqrt_dt = np.sqrt(dt)

    X_paths = np.empty((N, M + 1))
    X_paths[:, 0] = X0

    S_paths = np.empty_like(X_paths)
    S_paths[:, 0] = np.exp(X0)

    W_T = np.zeros(N)
    for m in range(M):
        t_m = m * dt

        dW = np.random.normal(
            loc=0.0,
            scale=sqrt_dt,
            size=N,
        )

        W_T += dW

        X_curr = X_paths[:, m]

        X_paths[:, m]

        X_paths[:, m + 1] = (
            X_curr + (kappa * (seasonal_mean(t_m) - X_curr) + sigma * theta) * dt + sigma * dW
        )
        S_paths[:, m + 1] = np.exp(X_paths[:, m + 1])

    payoffs = power_cap_payoff(
        S_paths=S_paths,
        K=K,
        delivery_indices=delivery_indices,
        r=r,
        T=T,
        M=M,
        volume=volume,
    )

    likelihood_ratio = np.exp(
        -theta * W_T - 0.5 * theta**2 * T
    )
    
    weighted_payoffs = payoffs * likelihood_ratio

    estimate = np.mean(weighted_payoffs)
    std = np.std(weighted_payoffs, ddof=1)

    positive_paths = np.sum(payoffs > 0)

    return estimate, std, positive_paths