# analytical pricing formulas for reference

from scipy.stats import norm
import numpy as np

def black_scholes_call_price(S0, K, r, sigma, T):
    if T <= 0:
        return max(S0 - K, 0.0)
    d1 = (
        np.log(S0 / K) + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    bs_call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return bs_call_price
