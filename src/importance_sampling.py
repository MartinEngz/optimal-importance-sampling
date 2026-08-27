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
    simulate_mean_reverting_jump_price_paths,
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


def evaluate_power_cap_is_theta(
        theta,
        dW,
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
        jump_increments=None,
        jump_intensity=None,
        jump_scale=None,
        jump_reversion=None
):
    """
    Evaluate fixed-control IS for a power cap using supplied Brownian 
    increments.

    Using the same dW for different candidate theta values gives
    common random numbers and makes the optimization objective
    substantially less noisy.
    """
    N = dW.shape[0]

    if jump_increments is None:
        # Diffusion-only model
        S_paths = simulate_mean_reverting_log_price_paths(
            N=N,
            X0=X0,
            kappa=kappa,
            sigma=sigma,
            T=T,
            M=M,
            seasonal_mean=seasonal_mean,
            theta=theta,
        )
    else:
        # Jump model
        S_paths = simulate_mean_reverting_jump_price_paths(
            N=N,
            X0=X0,
            kappa=kappa,
            sigma=sigma,
            T=T,
            M=M,
            seasonal_mean=seasonal_mean,
            jump_intensity=jump_intensity,
            jump_scale=jump_scale,
            jump_reversion=jump_reversion,
            theta=theta,
            dW=dW,
            jump_increments=jump_increments
        )

    payoffs = power_cap_payoff(
        S_paths=S_paths,
        K=K,
        delivery_indices=delivery_indices,
        r=r,
        T=T,
        M=M,
        volume=volume,
    )

    W_T = np.sum(dW, axis=1)

    log_L = (
        -theta * W_T - 0.5 * theta**2 * T
    )

    weighted_payoffs = (
        payoffs * np.exp(log_L)
    )

    estimate = np.mean(weighted_payoffs)
    std = np.std(weighted_payoffs, ddof=1)

    # Empirical second moment: minimizing this is equivalent to minimizing variance
    second_moment = np.mean(
        weighted_payoffs**2
    )

    positive_paths = np.sum(payoffs > 0)

    return {
        "estimate": estimate,
        "std": std,
        "second_moment": second_moment,
        "positive_paths": positive_paths,
        "positive_rate": positive_paths / N,
    }

def adaptive_is_theta_power_cap(
        X0, K, kappa, sigma, T, M, r, volume, delivery_indices, seasonal_mean,
        tol=0.05,
        N_bar=1_000,
        gamma=2.0,
        theta_min=0.0,
        theta_max=20.0,
        n_grid=41,
        n_refine=101,
        refine_half_width=1.0,
        max_iter=15,
        random_seed=42,
        jump_intensity=None,
        jump_scale=None,
        jump_reversion=None
):
    """
    Adaptive importance sampling for a cap-style power derivative.

    Iteratively minimizes the empirical second moment of the 
    importance-sampling estimator over a constant Brownian control theta.
    """
    rng = np.random.default_rng(random_seed)

    dt = T / M

    theta_previous = None

    history = []

    N = N_bar

    dW = rng.normal(
        loc=0.0,
        scale=np.sqrt(dt),
        size=(N_bar, M)
    )

    jump_counts = rng.poisson(
        jump_intensity * dt,
        size=(N_bar, M)
    )

    jump_increments = np.zeros(
        (N_bar, M)
    )

    mask = jump_counts > 0

    jump_increments[mask] = rng.gamma(
        shape=jump_counts[mask],
        scale=jump_scale
    )

    for iteration in range(1, max_iter + 1):
        N = dW.shape[0]
        
        # 1. Coarse search
        theta_grid = np.linspace(
            theta_min,
            theta_max, 
            n_grid
        )

        coarse_objective = []

        for theta in theta_grid:
            result = evaluate_power_cap_is_theta(
                theta=theta,
                dW=dW,
                X0=X0,
                K=K,
                kappa=kappa,
                sigma=sigma,
                T=T,
                M=M,
                r=r,
                volume=volume,
                delivery_indices=delivery_indices,
                seasonal_mean=seasonal_mean,
                jump_increments=jump_increments,
                jump_intensity=jump_intensity,
                jump_scale=jump_scale,
                jump_reversion=jump_reversion
            )

            coarse_objective.append(
                result["second_moment"]
            )

        coarse_objective = np.asarray(
            coarse_objective
        )

        best_coarse = theta_grid[
            np.argmin(coarse_objective)
        ]

        # 2. Local refinement
        lo = max(theta_min, best_coarse - refine_half_width)
        hi = min(theta_max, best_coarse + refine_half_width)

        theta_refined = np.linspace(lo, hi, n_refine)

        refined_objective = []

        for theta in theta_refined:
            result = evaluate_power_cap_is_theta(
                theta=theta,
                dW=dW,
                X0=X0,
                K=K,
                kappa=kappa,
                sigma=sigma,
                T=T,
                M=M,
                r=r,
                volume=volume,
                delivery_indices=delivery_indices,
                seasonal_mean=seasonal_mean,
                jump_increments=jump_increments,
                jump_intensity=jump_intensity,
                jump_scale=jump_scale,
                jump_reversion=jump_reversion
            )

            refined_objective.append(result["second_moment"])

        refined_objective = np.asarray(refined_objective)

        theta_opt = theta_refined[np.argmin(refined_objective)]

        # Diagnostics
        history.append({
            "iteration": iteration,
            "N": N,
            "theta": theta_opt,
            "objective": np.min(refined_objective),
        })

        print(
            f"Iteration {iteration:>2d} | "
            f"N={N:>7,d} | "
            f"dW.shape={dW.shape} | "
            f"theta_opt={theta_opt:.4f} | "
            f"J(theta)={np.min(refined_objective):.6e}"
        )

        # 3. Convergence test
        if theta_previous is not None:
            if abs(theta_opt - theta_previous) < tol:
                break

        theta_previous = theta_opt

        if iteration < max_iter:            
            N_new = int(np.ceil(gamma*N))

            n_extra = N_new - N

            extra_dW = rng.normal(
                loc=0.0,
                scale=np.sqrt(dt),
                size=(n_extra, M)
            )

            extra_counts = rng.poisson(
                jump_intensity * dt,
                size=(n_extra, M),
            )

            extra_jumps = np.zeros(
                (n_extra, M)
            )

            mask = extra_counts > 0

            extra_jumps[mask] = rng.gamma(
                shape=extra_counts[mask],
                scale=jump_scale,
            )

            dW = np.vstack((dW, extra_dW))

            jump_increments = np.vstack((jump_increments, extra_jumps))
            

    # Final independent simulation at selected optimal theta
    dW_final = rng.normal(
        loc=0.0,
        scale=np.sqrt(dt),
        size=(N,M),
    )

    jump_counts_final = rng.poisson(
            jump_intensity * dt,
            size=(N, M)
        )
    
    jump_increments_final = np.zeros(
        (N, M)
    )

    mask_final = jump_counts_final > 0

    jump_increments_final[mask_final] = rng.gamma(
        shape=jump_counts_final[mask_final],
        scale=jump_scale
    )

    final_result = evaluate_power_cap_is_theta(
        theta=theta_opt,
        dW=dW_final,
        X0=X0,
        K=K,
        kappa=kappa,
        sigma=sigma,
        T=T,
        M=M,
        r=r,
        volume=volume,
        delivery_indices=delivery_indices,
        seasonal_mean=seasonal_mean,
        jump_increments=jump_increments_final,
        jump_intensity=jump_intensity,
        jump_scale=jump_scale,
        jump_reversion=jump_reversion
    )

    return {
        "theta": theta_opt,
        "estimate": final_result["estimate"],
        "std": final_result["std"],
        "positive_paths": final_result["positive_paths"],
        "positive_rate": final_result["positive_rate"],
        "second_moment": final_result["second_moment"],
        "iterations": iteration,
        "N": N,
        "history": history,
        "theta_grid": theta_refined,
        "objective_grid": refined_objective,
    }