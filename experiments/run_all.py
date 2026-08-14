import sys

from experiments.crude_mc import (
    run_european_call as run_crude_european,
    run_barrier_call as run_crude_barrier,
)

from experiments.fixed_is import (
    run_european_call as run_fixed_is_european,
    run_barrier_call as run_fixed_is_barrier,
)

from experiments.adaptive_is import (
    run_european_call as run_adaptive_european
)

from experiments.optimal_control import (
    run_european_call as run_optimal_control_european,
    run_barrier_call as run_optimal_control_barrier,
)

def run_vanilla():
    print("\n=== EUROPEAN CALL ===")

    print("\n=== CRUDE MONTE CARLO ===")
    crude = run_crude_european()

    std_reference = crude["std_reference"]

    print("\n=== FIXED IMPORTANCE SAMPLING ===")
    fixed_is = run_fixed_is_european(
        std_reference=std_reference
    )

    print("\n=== ADAPTIVE IMPORTANCE SAMPLING ===")
    adaptive_is = run_adaptive_european(std_reference=std_reference)

    print("\n=== PDE_BASED OPTIMAL CONTROL ===")
    optimal_control = run_optimal_control_european(
        std_reference=std_reference
    )

    return {
        "crude_mc": crude,
        "fixed_is": fixed_is,
        "adaptive_is": adaptive_is,
        "optimal_control": optimal_control
    }

def run_barrier():
    print("\n=== UP-AND-OUT BARRIER CALL ===")

    print("\n=== CRUDE MONTE CARLO ===")
    crude = run_crude_barrier()

    std_reference = crude["std_reference"]

    print("\n=== FIXED IMPORTANCE SAMPLING ===")
    fixed_is = run_fixed_is_barrier(
        std_reference=std_reference
    )

    print("\n=== PDE-BASED OPTIMAL CONTROL ===")
    optimal_control = run_optimal_control_barrier(
        std_reference=std_reference
    )

    return {
        "crude_mc": crude,
        "fixed_is": fixed_is,
        "optimal_control": optimal_control,
    }

def main(experiment="vanilla"):
    if experiment == "vanilla":
        return run_vanilla()

    elif experiment == "barrier":
        return run_barrier()

    elif experiment == "all":
        return {
            "vanilla": run_vanilla(),
            "barrier": run_barrier(),
        }
    else:
        raise ValueError(
            "Unknown experiment. "
            "Choose 'vanilla', 'barrier', or 'all'."
        )

if __name__ == "__main__":
    """"
    CLI options:
        python3 -m experiments.run_all vanilla
        python3 -m experiments.run_all barrier
        python3 -m experiments.run_all all
    """

    experiment = (
        sys.argv[1].lower()
        if len(sys.argv) > 1
        else "vanilla"
    )

    main(experiment)