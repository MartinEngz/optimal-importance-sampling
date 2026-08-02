# European call, barrier call
import numpy as np

def european_call_payoff(S_T, K, r, T):
    return np.exp(-r * T) * np.maximum(S_T - K, 0.0)

def up_and_out_call_payoff(S_paths, K, r, T, U):
    S_T = S_paths[:, -1]
    running_max = np.max(S_paths, axis=1)

    payoff = np.exp(-r * T) * np.maximum(S_T - K, 0.0)
    return np.where(running_max <= U, payoff, 0.0)