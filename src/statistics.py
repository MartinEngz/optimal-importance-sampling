import numpy as np
from scipy.stats import norm

def standard_error(std, N):
    return std / np.sqrt(N)

def confidence_interval(estimate, std, N, alpha=0.05):
    z = norm.ppf(1 - alpha / 2)
    se = standard_error(std, N)

    lower = estimate - z * se
    upper = estimate + z * se

    return lower, upper

def confidence_interval_length(estimate, std, N, alpha=0.05):
    lower, upper = confidence_interval(estimate, std, N, alpha(0.05))
    return upper - lower

def variance_reduction(std_reference, std_method):
    return (std_reference / std_method) ** 2

def absolute_error(estimate, reference):
    return abs(estimate - reference)