from experiments.crude_mc import run_european_call as run_crude
from experiments.fixed_is import run_european_call as run_fixed_is
from experiments.adaptive_is import run_european_call as run_adaptive_is
from experiments.optimal_control import run_european_call as run_optimal_control

def main():
    print("\n=== CRUDE MONTE CARLO ===")
    crude = run_crude()

    std_reference = crude["std_reference"]

    print("\n=== FIXED IMPORTANCE SAMPLING ===")
    fixed_is = run_fixed_is(std_reference=std_reference)

    print("\n=== ADAPTIVE IMPORTANCE SAMPLING ===")
    adaptive_is = run_adaptive_is(std_reference=std_reference)

    print("\n=== PDE_BASED OPTIMAL CONTROL ===")
    optimal = run_optimal_control(std_reference=std_reference)

    return {
        "crude_mc": crude,
        "fixed_is": fixed_is,
        "adaptive_is": adaptive_is,
        "optimal_control": optimal
    }

if __name__ == "__main__":
    main()