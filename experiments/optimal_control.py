import time
import numpy as np

from experiments.config import (
    EUROPEAN_CALL, 
    OPTIMAL_CONTROL, 
    PDE_REFINEMENT,
    BARRIER_CALL,
    BARRIER_OPTIMAL_CONTROL,
)

from src.pricing import black_scholes_call_price

from src.pde import (
    solve_european_call_kbe,
    solve_barrier_call_kbe,
)

from src.optimal_control import (
    compute_optimal_control,
    get_control_interpolator,
    simulate_controlled_european_call,
    simulate_controlled_up_and_out_call,
)

from src.statistics import variance_reduction

from src.plotting import (
    plot_vrf_vs_pde_refinement,
    plot_barrier_optimal_control,
    plot_controlled_barrier_paths,
)

def run_european_call(std_reference=None):
    """
    1. solve KBE
    2. compute/interpolate zeta*
    3. simulate controlled paths
    4. calculate estimator variance
    5. compare against crude MC
    6. run PDE grid refinement study
    """

    cfg=EUROPEAN_CALL
    optimal_control_cfg=OPTIMAL_CONTROL

    S0 = cfg["S0"]
    K = cfg["K"]
    r = cfg["r"]
    sigma = cfg["sigma"]
    T = cfg["T"]

    M_pde = optimal_control_cfg["M_pde"]
    P_pde = optimal_control_cfg["P_pde"]
    N_sim = optimal_control_cfg["N_sim"]

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

    est, std = simulate_controlled_european_call(
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
    print(f"IS Estimate (Z_dt):    {est:.6f}")
    print(f"IS std:           {std:.4e}")
    
    if std_reference is not None:
        vrf = variance_reduction(
            std_reference,
            std
        )
        print(f"\nVariance reduction vs CMC: {vrf:.2f}")


    run_refinement = PDE_REFINEMENT["VRF_vs_discretization_analysis"]

    full_results = {}
    vrf_results = {}

    if std_reference is not None and run_refinement:
        print("\n--- VRF vs PDE discretization couples ---")

        N_test = PDE_REFINEMENT["N_test"]
        P_list = PDE_REFINEMENT["P_list"]
        M_list = PDE_REFINEMENT["M_list"]

        for m_val in M_list:
            for p_val in P_list:
                start_time = time.time()
                s_grid, t_grid, v = solve_european_call_kbe(S0=S0, K=K, T=T, r=r, sigma=sigma, M=m_val, P=p_val)
                zeta_star = compute_optimal_control(v=v, s_grid=s_grid, sigma=sigma)
                ctrl_func = get_control_interpolator(t_grid=t_grid, s_grid=s_grid, zeta=zeta_star)
                est, std = simulate_controlled_european_call(N=N_test, S0=S0, K=K, T=T, r=r, sigma=sigma, M=m_val, control_func=ctrl_func)
                
                end_time = time.time()
                elapsed = end_time - start_time

                vrf = variance_reduction(
                    std_reference,
                    std
                )
                full_results[(p_val, m_val)] = {
                    'price': est,
                    'std': std,
                    'vrf': vrf,
                    'time': elapsed
                }
                vrf_results[(p_val, m_val)] = vrf
                
                print(f"(P={p_val}, M={m_val}) | Price: {est:.8f} | std: {std:.8f} | VRF: {vrf_results[(p_val, m_val)]:.2f} | Time: {elapsed:.2f}s")

        plot_vrf_vs_pde_refinement(M_list, P_list, full_results)

    return {
        "estimate": est,
        "std": std,
        "bs_price": bs_price,
        "full_results": full_results,
        "vrf_results": vrf_results
    }

def run_barrier_call(std_reference=None):
    cfg = BARRIER_CALL
    oc_cfg = BARRIER_OPTIMAL_CONTROL

    # Barrier contract / SDE parameters
    S0 = cfg["S0"]
    K = cfg["K"]
    U = cfg["U"]
    r = cfg["r"]
    sigma = cfg["sigma"]
    T = cfg["T"]
    M_sim = cfg["M"]

    # PDE parameters
    P_pde = oc_cfg["P_pde"]
    M_pde = oc_cfg["M_pde"]
    epsilon_bc = oc_cfg["epsilon_bc"]
    N_sim = oc_cfg["N_paths_plot"]

    # --- 1. Solve barrier KBE
    s_grid, t_grid, v = solve_barrier_call_kbe(
        S0=S0,
        K=K,
        U=U,
        T=T,
        r=r,
        sigma=sigma,
        M=M_pde,
        P=P_pde,
        epsilon_bc=epsilon_bc,
    )

    # --- 2. Construct optimal control
    zeta = compute_optimal_control(
        v=v,
        s_grid=s_grid,
        sigma=sigma
    )

    control = get_control_interpolator(
        t_grid=t_grid,
        s_grid=s_grid,
        zeta=zeta
    )

    # --- 3. Plot the control
    plot_barrier_optimal_control(
        t_grid=t_grid,
        s_grid=s_grid,
        zeta=zeta,
        K=K,
        U=U,
    )

    # --- 4. Visualize 100 controlled paths
    _, _, path_diagnostics, paths = simulate_controlled_up_and_out_call(
        N=N_sim,
        S0=S0,
        K=K,
        U=U,
        T=T,
        r=r,
        sigma=sigma,
        M=M_sim,
        control_func=control,
        return_paths=True
    )

    plot_controlled_barrier_paths(
        paths=paths,
        T=T,
        K=K,
        U=U
    )

    # --- 5. Main pricing run
    est, std, diagnostics, _ = simulate_controlled_up_and_out_call(
        N=N_sim,
        S0=S0,
        K=K,
        U=U,
        T=T,
        r=r,
        sigma=sigma,
        M=M_sim,
        control_func=control,
        return_paths=False
    )

    print("-" * 45)
    print("Up-and-out call: optimal-control IS")
    print(
        f"PDE grid: M={M_pde}, P={P_pde} | "
        f"Simulation: M={M_sim}, N={N_sim}"
    )
    print("-" * 45)

    print(f"OC-IS estimate:     {est:.8f}")
    print(f"OC-IS std:          {std:.8f}")

    print(
        f"Terminal > K:     "
        f"{diagnostics['over_strike']:,}"
    )

    print(
        f"Barrier hits:     "
        f"{diagnostics['barrier_hits']:,}"
    )

    vrf = None

    if std_reference is not None:
        vrf = variance_reduction(
            std_reference=std_reference,
            std_method=std
        )
        print(
            f"Variance reduction vs CMC: "
            f"{vrf:.2f}"
        )

    return {
        "estimate": est,
        "std": std,
        "variance_reduction": vrf,
        "diagnostics": diagnostics,
        "zeta": zeta,
        "s_grid": s_grid,
        "t_grid": t_grid,
    }


if __name__ == "__main__":
    # run_european_call()
    run_barrier_call()