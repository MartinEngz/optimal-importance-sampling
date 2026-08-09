EUROPEAN_CALL = {
    "S0": 100.0,
    "K": 120.0,
    "r": 0.05,
    "sigma": 0.51,
    "T": 0.2,
    "U": 200.0,
    "M": 100
}

N_LIST = [
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000
]

ADAPTIVE_IS = {
    "tol": 5e-2,
    "alpha": 0.05,
    "N_bar": 500,
    "gamma": 2.0,
    "r_min": 0.05,
    "r_max": 5.0,
    "n_grid": 100,
    "n_refine": 501,
    "refine_half_width": 0.25,
    "max_iter": 20,
    "random_seed":123
}

# Add BARRIER_CALL and PDE grids here later