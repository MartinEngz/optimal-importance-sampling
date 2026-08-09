from experiments.config import EUROPEAN_CALL
from src.pde import solve_european_call_kbe
from src.optimal_control import (
    make_control_interpolator,
    simulate_controlled_gbm
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

    return {}

if __name__ == "__main__":
    run_european_call()