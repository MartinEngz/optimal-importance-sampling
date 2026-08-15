# European call, barrier call
import numpy as np

def european_call_payoff(S_T, K, r, T):
    return np.exp(-r * T) * np.maximum(S_T - K, 0.0)

def up_and_out_call_payoff(S_paths, K, r, T, U):
    S_T = S_paths[:, -1]
    running_max = np.max(S_paths, axis=1)

    vanilla_payoff = np.exp(-r * T) * np.maximum(S_T - K, 0.0)
    return np.where(running_max <= U, vanilla_payoff, 0.0)

def power_cap_payoff(
    S_paths,
    K,
    delivery_indices,
    r,
    T,
    M,
    volume=1.0, # e.g 1 MW cont. delivered over each interval
):
    """
    Cap-style electricity payoff over multiple delivery periods.

    payoff = volume * sum_i exp(-r t_i) * max(S_{t_i}, 0) * Delta_t
    """
    dt = T / M

    delivery_prices = S_paths[:, delivery_indices]
    delivery_times = (
        np.asarray(delivery_indices) * dt
    )

    discount_factors = np.exp(
        -r * delivery_times
    )

    caplets = np.maximum(
        delivery_prices - K, 
        0.0
    )

    return (
        volume * np.sum(caplets * discount_factors[None, :] * dt, axis=1)
    )