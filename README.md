# Optimal Importance Sampling

Numerical methods for variance reduction in Monte Carlo derivative pricing using importance sampling and stochastic optimal control.

The project compares:

- Crude Monte Carlo simulation

- Parametric importance sampling through drift modification

- Adaptive variance-minimizing importance sampling

- PDE-based optimal control importance sampling

The methodology is first studied for European and up-and-out call options. The optimal state-dependent change of measure is obtained from a numerical solution of the backward Kolmogorov equation and used to steer simulated paths toward payoff-relevant regions.

The original project was developed for the EPFL course *Stochastic Simulation (MATH-414)* and is being rewritten as a standalone numerical finance project, with a planned extension to rare-event simulation in energy and commodity derivatives.

## Why this project?

Rare-event payoffs can make crude Monte Carlo highly inefficient. This project investigates how changing the simulation measure can dramatically reduce estimator variance while preserving the desired expectation.