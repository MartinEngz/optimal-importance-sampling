import time
import numpy as np

from experiments.config import EUROPEAN_CALL
from src.pricing import black_scholes_call_price
from src.pde import solve_european_call_kbe
from src.optimal_control import (
    compute_optimal_control,
    get_control_interpolator,
    simulate_controlled_european_call
)
from src.statistics import variance_reduction

def run_european_call(std_reference=None):
    cfg=EUROPEAN_CALL

    # 1. solve KBE
    # 2. compute/interpolate zeta*
    # 3. simulate controlled paths
    # 4. calculate estimator variance
    # 5. compare against crude MC
    # 6. run PDE grid refinement study
    # 7. generate control/convergence plot

    S0 = cfg["S0"]
    K = cfg["K"]
    r = cfg["r"]
    sigma = cfg["sigma"]
    T = cfg["T"]

    M_pde = 30
    P_pde = 50
    N_sim = 1_000_000

    bs_price = black_scholes_call_price(
        S0, K, r, sigma, T
    )

    s_grid, t_grid, v = solve_european_call_kbe(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        M=M_pde,
        P=P_pde
    )

    zeta = compute_optimal_control(
        v=v,
        s_grid=s_grid,
        sigma=sigma
    )

    control = get_control_interpolator(
        t_grid,
        s_grid,
        zeta
    )

    Z_is, Var_is, error_is = simulate_controlled_european_call(
        N = N_sim,
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        M=M_pde,
        control_func=control
    )
    
    print("-" * 30)
    print(f"Numerical Setup: M={M_pde}, P={P_pde}, N={N_sim}")
    # print(f"S1: {S1:.2f}, S2: {S2:.2f}")
    print("-" * 30)
    print(f"Closed-form BS Price:  {bs_price:.6f}")
    print(f"IS Estimate (Z_dt):    {Z_is:.6f}")
    print(f"Statistical Error:     {error_is:.6f}")
    print(f"IS Variance:           {Var_is:.4e}")
    print(f"Relative Bias (%):     {abs(Z_is - bs_price)/bs_price * 100:.4f}%")
    print(f"Absolute Bias (%):     {abs(Z_is - bs_price) * 100:.4f}%")
    print("-" * 30)

    if std_reference is not None:
        std_is = np.sqrt(Var_is)
        vrf = variance_reduction(
            std_reference,
            std_is
        )
        print(f"\nVariance reduction vs CMC: {vrf:.2f}")


    return {}

if __name__ == "__main__":
    run_european_call()