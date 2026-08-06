import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from src.pricing import black_scholes_call_price
from src.monte_carlo import mc_european_call_em


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
# We choose the IS drift that centers log-price at log(K)
r_tilde = np.log(K/S0) / T + 0.5 * sigma**2 