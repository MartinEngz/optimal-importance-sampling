# Optimal Importance Sampling

A modular rework of an EPFL **Stochastic Simulation** project on variance reduction for option pricing.

The project studies how importance sampling can improve Monte Carlo estimation when option payoffs are driven by rare events. It progresses from standard crude Monte Carlo to fixed-drift importance sampling, adaptive importance sampling, and PDE-based optimal control.

The current implementation covers:

- a European call option under Black-Scholes dynamics;
- an up-and-out European barrier call;
- crude Monte Carlo benchmarks;
- fixed-drift importance sampling;
- adaptive importance sampling for the vanilla European call;
- PDE-based state-dependent optimal control;
- variance-reduction and convergence diagnostics.

A later extension will apply the same rare-event simulation ideas to **Nordic power markets**, using commodity-specific price dynamics and a cap-style high-price payoff.

## Project structure

```text
optimal-importance-sampling/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── models.py
│   ├── payoffs.py
│   ├── pricing.py
│   ├── monte_carlo.py
│   ├── importance_sampling.py
│   ├── pde.py
│   ├── optimal_control.py
│   ├── statistics.py
│   └── plotting.py
└── experiments/
    ├── __init__.py
    ├── config.py
    ├── crude_mc.py
    ├── fixed_is.py
    ├── adaptive_is.py
    ├── optimal_control.py
    └── run_all.py
```

The `src/` package contains reusable numerical methods, while `experiments/` defines the numerical studies, parameter choices, comparisons, and plots.

## 1. European call

The first part considers a European call under geometric Brownian motion,

$$
dS_t = rS_t dt + \sigma S_t dW_t,
$$

with payoff

$$
e^{-rT}(S_T-K)^+.
$$

The Black-Scholes price is used as an analytical benchmark.

### Crude Monte Carlo

The baseline estimator is

$$
\hat{V}_N = \frac{1}{N} \sum_{i=1}^N e^{-rT}(S_T^{(i)}-K)^+.
$$

For an out-of-the-money option, many simulated paths produce zero payoff, making crude Monte Carlo inefficient at small sample sizes.

### Fixed-drift importance sampling

A first importance-sampling scheme changes the simulated drift from $r$ to a fixed $\tilde r$. The likelihood ratio corrects for the change of measure so that the estimator remains unbiased.

This increases the frequency of payoff-relevant paths and reduces estimator variance.

### Adaptive importance sampling

The fixed drift is then selected numerically by minimizing an empirical second-moment objective. This improves on the heuristic drift choice while remaining within a one-parameter family of changes of measure.

### PDE-based optimal control

The final vanilla method uses a state- and time-dependent control

$$\zeta^*(t,s) = \sigma s \frac{\partial_s v(t,s)}{v(t,s)}$$

where $v(t,s)$ is obtained by solving the Kolmogorov backward equation numerically.

Controlled paths are simulated under

$$ dS_t^\zeta = \left(rS_t^\zeta + \sigma S_t^\zeta \zeta_t\right)dt + \sigma S_t^\zeta dW_t $$

with the corresponding Girsanov likelihood ratio.

This demonstrates how a numerically computed control can strongly reduce variance relative to crude Monte Carlo.

## 2. Up-and-out barrier call

The second part extends the framework to the path-dependent payoff

$$ e^{-rT}(S_T-K)^+ \mathbf{1} \left\( \max_{0\le t\le T} S_t \le U \right). $$

This creates a more difficult rare-event problem: simulated paths should finish above the strike while avoiding the upper barrier.

### Crude Monte Carlo

For the chosen parameters, positive payoffs are rare. At low sample sizes, crude Monte Carlo may observe no payoff-producing paths at all.

### Fixed-drift importance sampling

Increasing the drift makes terminal prices above the strike more common, but it also increases the number of barrier knock-outs.

The barrier experiment therefore exposes a structural limitation of a single constant drift:

$$
\text{push toward } K
\qquad \text{vs.} \qquad
\text{avoid } U.
$$

A strike-versus-barrier diagnostic plot is used to visualize this trade-off.

### State-dependent optimal control

The barrier Kolmogorov backward equation is solved on a domain with upper boundary $S_{\max}=U$. The same reusable control construction as in the European-call case is then applied.

The resulting optimal control is state dependent:

- it pushes paths upward when prices are too low;
- it weakens or reverses the push near the barrier.

Controlled-path plots illustrate how the importance-sampling measure concentrates simulations in the payoff-relevant region.

Adaptive constant-drift importance sampling is intentionally omitted for the barrier case: optimizing within the same one-parameter constant-drift family does not resolve the strike/barrier conflict revealed by the fixed-drift experiment.

## Running the experiments

From the repository root, run:

```bash
python3 -m experiments.run_all vanilla
```

for the European-call experiments,

```bash
python3 -m experiments.run_all barrier
```

for the barrier-call experiments, or

```bash
python3 -m experiments.run_all all
```

to run both.

Individual experiment modules can also be executed directly, for example:

```bash
python3 -m experiments.crude_mc
python3 -m experiments.fixed_is
python3 -m experiments.adaptive_is
python3 -m experiments.optimal_control
```

## Main numerical ideas

The project focuses on:

- Monte Carlo estimation;
- rare-event simulation;
- importance sampling;
- likelihood-ratio / Girsanov reweighting;
- empirical variance minimization;
- finite-difference PDE methods;
- Kolmogorov backward equations;
- stochastic optimal control;
- variance-reduction factors;
- numerical convergence diagnostics.

## Next step: Nordic power extension

The next phase moves beyond the original EPFL assignment and applies the same methodology to a stylized Nordic electricity-market problem.

The planned extension is a **Nordic-power-inspired cap-style derivative** with payoff exposure over multiple delivery periods, combined with electricity-specific spot dynamics such as:

- mean reversion;
- deterministic seasonality;
- price spikes / jumps.

A baseline specification will use a mean-reverting log-price process,

$$ dX_t = \kappa\bigl(\mu(t)-X_t\bigr)dt + \sigma dW_t \qquad S_t=e^{X_t} $$

and a cap-style payoff of the form

$$ Y = \sum_i e^{-rt_i} (S_{t_i}-K)^+\Delta t.$$

The goal is to study whether importance sampling can materially reduce the computational cost of estimating high-price electricity exposures driven by rare market events.

## Origin

This repository is based on a project completed in **EPFL MATH-414 Stochastic Simulation** and has been rewritten into a modular Python codebase.

The rework emphasizes:

- reusable numerical components;
- clear separation between algorithms and experiments;
- more efficient path handling;
- cleaner diagnostics;
- a direct path toward an energy-market application.

## Keywords

**Keywords:** Monte Carlo, importance sampling, rare-event simulation, variance reduction, optimal control, Girsanov theorem, Kolmogorov backward equation, finite differences, option pricing, barrier options, stochastic simulation, quantitative finance, energy derivatives, electricity markets, Nordic power
