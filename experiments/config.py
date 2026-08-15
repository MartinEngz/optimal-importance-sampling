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

OPTIMAL_CONTROL = {
    "M_pde": 30,
    "P_pde": 50,
    "N_sim": 1_000_000
}

PDE_REFINEMENT = {
    "VRF_vs_discretization_analysis": False,
    "N_test": 100_000,
    "P_list": [20, 60, 100, 300, 500],
    "M_list": [10, 50, 100, 500, 2000, 5000, 10_000]
}

BARRIER_CALL = {
    "S0": 100.0,
    "K": 150.0,
    "r": 0.10,
    "sigma": 0.30,
    "T": 0.20,
    "U": 200.0,
    "M": 1000,          # Euler-Maruyama steps for simulated paths
}

BARRIER_OPTIMAL_CONTROL = {
    "P_pde": 200,
    "M_pde": 200,
    "epsilon_bc": 0.1,
    "N_paths_plot": 100,
}

BARRIER_FINE = {
    "U": 170.0,
    "P_pde": 3000,
    "M_pde": 1500,
}

POWER_CAP = { # intial numerical parameters, not a Nord Pool calibration
    "S0": 50.0,          # EUR/MWh
    "base_price": 50.0,
    "K": 100.0,          # high strike
    "kappa": 8.0,
    "sigma": 0.8,
    "T": 30 / 365,       # one-month horizon
    "M": 30,             # initial daily discretization
    "r": 0.02,
    "volume": 1.0,
    "seasonal_amplitude": 0.20,
    "seasonal_phase": 0.0,
}

POWER_CAP_N_LIST = [
    100,
    1_000,
    10_000,
    100_000,
    1_000_000
]