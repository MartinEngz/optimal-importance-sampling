import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from src.pricing import black_scholes_call_price
from src.monte_carlo import mc_european_call_em
from src.importance_sampling import is_fixed_rtilde, adaptive_is_rtilde
from src.models import log_price_params


np.random.seed(42) # for reproducibility

# --- Experiment parameters (adapted from EPFL Stochastic Simulation project) ---
r = 0.05
sigma = 0.51
T = 0.2
S0 = 100.0
K = 120.0

M_em = 100 # number of EM (Euler-Maruyama steps)
# dt_em = T / M_em

# --- Exact bs_call_price for reference ---
bs_price = black_scholes_call_price(S0, K, r, sigma, T)
print(f"Reference Black-Scholes price: {bs_price}")


# --- Crude Monte Carlo Method
# Example run of crude MC for a range of N
N_list = [10, 100, 1000, 10_000, 100_000, 1_000_000]

cmc_results = []
for N in N_list:
    est, est_std = mc_european_call_em(
        N=N, 
        S0=S0, 
        r=r, 
        sigma=sigma, 
        T=T, 
        M=M_em, 
        K=K)
    cmc_results.append((N, est, est_std))

var_crude_ref = cmc_results[-1][2]**2
print(f"Variance of crude MC estimator with N={N_list[-1]}: {var_crude_ref}")


# --- Prepare data for plots ---
# Extract arrays from cmc_results = [(N, est, std_payoff), ...]
N_arr = np.array([res[0] for res in cmc_results], dtype=float)
est_arr = np.array([res[1] for res in cmc_results], dtype=float)
std_payoff_arr = np.array([res[2] for res in cmc_results], dtype=float)

# Standard error of the crude Monte Carlo estimator
se_arr = std_payoff_arr / np.sqrt(N_arr)


# --- Plot 1: Convergence rate of the standard error (log-log)

plt.figure(figsize=(8.5, 4.8))

#Empirical standard error
plt.loglog(
    N_arr, se_arr,
    'o-',
    linewidth=1.2,
    markersize=5,
    label="Empirical SE"
)

# Reference slope ~ 1/sqrt(N)
ref_se = se_arr[0] * np.sqrt(N_arr[0]) / np.sqrt(N_arr)
plt.loglog(
    N_arr, ref_se,
    '--',
    linewidth=1.2,
    label=r"Reference $\propto N^{-1/2}$"
)

plt.xlabel("Number of samples (N)")
plt.ylabel("Standard error")
plt.title("Crude Monte Carlo: convergence rate of standard error")

plt.grid(True, which='both', linestyle=':', linewidth=0.8)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 2: MC price estimates with 95% CI and analytical reference

plt.figure(figsize=(8.5, 4.8))

# Quantile for asymptotic 95% confidence intervals
z975 = norm.ppf(0.975)

# 95% CI half-widths
yerr = z975 * se_arr

# Crude MC estimates with CIs
plt.errorbar(
    N_arr,
    est_arr,
    yerr=yerr,
    fmt='o-',
    markersize=5,
    linewidth=1.2,
    capsize=4,
    capthick=1.0,
    elinewidth=1.0,
    label="CMC Estimate"
)

# Analytical Black-Scholes reference
plt.axhline(
    bs_price,
    color='red',
    linestyle='--',
    linewidth=1.5,
    label="Analytical Black-Scholes"
)

plt.xscale('log')
plt.xlabel("Number of samples (N)")
plt.ylabel("Call Price Estimate")
plt.title("European Call via Crude Monte Carlo")

plt.grid(True, which='both', linestyle=':', linewidth=0.8)
plt.legend()
plt.tight_layout()
plt.show()

# Print numerical 95% CIs for IS estimates
print("\n95% Condidence Intervals for Crude Monte Carlo estimates:\n")

for N, est, std in cmc_results:
    se = std / np.sqrt(N)
    ci_low = est - z975 * se
    ci_high = est + z975 * se
    ci_length = ci_high - ci_low

    print(
        f"N = {N:>7d} | "
        f"Estimate = {est:.8f} | "
        f"95% CI = [{ci_low:.8f}, {ci_high:.8f}] | "
        f"CI length = {ci_length:.8e} | "
        f"Std = {std:.8f}"
    )

# --- European call by importance sampling with modified drift ---

# We choose the IS drift such that log-price is centered at log(K)
r_tilde = np.log(K/S0) / T + 0.5 * sigma**2
print(f"Using r_tilde = {r_tilde:.4f} for importance sampling. \n")

is_results_fixed = []
for N in N_list:
    est, std = is_fixed_rtilde(N, S0, K, r, r_tilde, sigma, T)
    is_results_fixed.append((N, est, std))

print("\n95% Confidence Intervals for Fixed Importance Sampling estimates:\n")

for N, est, std in is_results_fixed:
    se = std / np.sqrt(N)
    ci_low = est - z975 * se
    ci_high = est + z975 * se
    ci_length = ci_high - ci_low

    print(
        f"N = {N:>7d} | "
        f"Estimate = {est:.8f} | "
        f"95% CI = [{ci_low:.8f}, {ci_high:.8f}] | "
        f"CI length = {ci_length:.8e} | "
        f"Std = {std:.8f}"
    )

print("\nAbsolute errors from Black-Scholes price:")
for res in is_results_fixed:
    print("\t For N = " + str(res[0]) + ": " + str(abs(res[1] - bs_price)))

# --- Quantification of Variance Reduction 
# --- Crude Monte Carlo vs Importance Sampling
VR_cmc_vs_is_fixed = []
for (N_cmc, _, std_cmc), (N_is, _, std_is) in zip(cmc_results, is_results_fixed):
    print(std_cmc, std_is)
    vr = (std_cmc ** 2) / (std_is ** 2)
    VR_cmc_vs_is_fixed.append((N_cmc, vr))

print(VR_cmc_vs_is_fixed)

# --- Plot: IS estimates with 95% CI + Analytical Black-Scholes line

# Extract arrays from is_results_fixed = [(N, est, std_weighted_payoff), ...]
N_is = np.array([res[0] for res in is_results_fixed], dtype=float)
est_is = np.array([res[1] for res in is_results_fixed], dtype=float)
std_is_payoff = np.array([res[2] for res in is_results_fixed], dtype=float)

# Standard error of the IS estimator
se_is = std_is_payoff / np.sqrt(N_is)

# 95% asymptotic CI half-width
yerr_is = z975 * se_is

plt.figure(figsize=(10, 5))

# Analytical BS reference
plt.axhline(bs_price, color='red', linestyle='--', linewidth=2, label='Analytical Black-Scholes')

# IS estimate with CI error bars
plt.errorbar(
    N_is, est_is,
    yerr=yerr_is,
    fmt='o-',
    markersize=6,
    linewidth=2,
    capsize=6,
    capthick=1.5,
    elinewidth=1.5,
    label='IS estimate'
)

plt.xscale('log')
plt.xlabel('Number of samples (N)')
plt.ylabel('Call Price Estimate')
plt.title('European Call via Importance Sampling')

plt.grid(True, which='both', linestyle=':', linewidth=1)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# --- Adaptive Importance Sampling with variance minimization ---
# Implementation of the adaptive importance sampling estimator as in Algorithm 6.4
# in the lecture notes to MATH-414.
adaptive_result = adaptive_is_rtilde(
    S0=S0, K=K, r=r, sigma=sigma, T=T,
    tol=5e-2, alpha=0.05,
    N_bar=500, gamma=2.0,
    r_min=0.05, r_max=5.0, n_grid=100,
    n_refine=501, refine_half_width=0.25,
    max_iter=20,
    random_seed=123
)

print("\nFinal adaptive results:")
print("  r_tilde_final =", adaptive_result["r_tilde_final"])
print("  N_final       =", adaptive_result["N_final"])
print("  Z_hat         =", adaptive_result["Z_hat"])
print("  sigma_hat     =", adaptive_result["sigma_hat"])

# --- Sanity check plot: J(theta) around the final minimizer ---

X = adaptive_result["last_X"]
payoffs = adaptive_result["last_payoffs"]
f_r_vals = adaptive_result["last_f_r_vals"]
f_rt_vals_current = adaptive_result["last_f_rt_vals"]

if X is None:
    print("\nNo sample stored for J(theta) sanity plot (algorithm may have stopped before sampling).")
else:
    r_tilde_final = adaptive_result["r_tilde_final"]
    r_min = adaptive_result["r_min"]
    r_max = adaptive_result["r_max"]
    hw = adaptive_result["refine_half_width"]
    n_refine = adaptive_result["n_refine"]
    var_X = adaptive_result["var_X"]
    std_X = np.sqrt(var_X)
    eps_denom = adaptive_result["eps_denom"]

    lo = max(r_min, r_tilde_final - hw)
    hi = min(r_max, r_tilde_final + hw)
    r_grid_plot = np.linspace(lo, hi, n_refine)

    f_r_sq = f_r_vals ** 2
    J_plot = np.empty_like(r_grid_plot, dtype=float)


    for j, r_candidate in enumerate(r_grid_plot):
        mu_cand, _ = log_price_params(r_candidate, sigma, T)
        f_cand = norm.pdf(X, loc=mu_cand, scale=std_X)
        denom = (f_cand * f_rt_vals_current) + eps_denom
        J_plot[j] = np.mean((payoffs ** 2) * f_r_sq / denom)

    plt.figure(figsize=(9, 4.8))
    plt.plot(r_grid_plot, J_plot, linewidth=2)
    plt.axvline(r_tilde_final, linestyle='--', linewidth=2, label=r"final $\tilde r$")
    plt.xlabel(r"$\theta$ (candidate $\tilde r$)")
    plt.ylabel(r"$\hat J(\theta)$")
    plt.title(r"Sanity check: objective $\hat J(\theta)$ around the minimizer")
    plt.grid(True, which='both', linestyle=':', linewidth=1)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Quantification of Variance Reduction - Crude Monte Carlo vs Adaptive Importance Sampling ---
print("Absolute errors from Black-Scholes price:")
print("\t For N = " + str(adaptive_result['N_final']))
print("Final standard deviation of weighted payoff: " + str(adaptive_result['sigma_hat']))
print("Variance Reduction: " + str((std_payoff_arr[-1] ** 2) / adaptive_result['sigma_hat'] ** 2))

## Comments on adaptive importance sampling
# Performance for tolerance (tol) = 1e-3:
# - Execution time: 6m1.0s
# - Iterations    : 18
# - r_tilde_final : 2.03
# - N_final       : 32768000
# - Z_hat         : 3.2182141337687673
# - sigma_hat     : 2.2164958762434037
# - Var reduction : 17.585903014316713

# Performance for tolerance (tol) = 5e-2:
# - Execution time: 0.1s
# - Iterations    : 6
# - r_tilde_final : 2.033
# - N_final       : 8000
# - Z_hat         : 3.1866129699491994
# - sigma_hat     : 2.2176522615636065
# - Var reduction : 17.56756760767114