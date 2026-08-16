# Optimal Importance Sampling for Energy Derivatives

A quantitative simulation project focused on **rare-event pricing and variance reduction in electricity markets**.

The project began as a modular rework of an EPFL **MATH-414 Stochastic Simulation** assignment on importance sampling for option pricing. The original European and barrier-option experiments now serve mainly as a theoretical and software foundation for the main extension: applying importance sampling to a stylized power-market derivative under electricity-specific price dynamics.

The current energy extension studies a **cap-style electricity derivative** under a seasonal mean-reverting log-price model. The central question is:

> Can importance sampling materially reduce the computational cost of estimating rare high-price electricity exposures?

So far, the answer is yes: the crude Monte Carlo problem is strongly rare-event dominated, while adaptive importance sampling increases the frequency of payoff-relevant paths from roughly **0.05% to about 50%** and reduces variance by **several hundred times**.

## Energy extension

### Mean-reverting power-price model

The baseline electricity model is

$$
dX_t = \kappa\bigl(\mu(t)-X_t\bigr)dt + \sigma dW_t,
$$

with

$$
S_t = e^{X_t}.
$$

Here:

- $S_t$ is the electricity spot price;
- $X_t$ is the log spot price;
- $\kappa$ controls mean-reversion speed;
- $\mu(t)$ is a deterministic seasonal mean;
- $\sigma$ controls short-term diffusion volatility.

The seasonal component is currently represented by a simple sinusoidal function. This is a deliberately stylized first model rather than a calibrated Nord Pool specification.

Mean reversion and seasonality are used because they are characteristic features of electricity spot-price modelling and provide a more realistic starting point than geometric Brownian motion.

### Cap-style multi-period payoff

Instead of a standard terminal European payoff, the energy extension uses exposure across a sequence of delivery dates:

$$
Y = q \sum_{i} e^{-rt_i} \left(S_{t_i}-K\right)^+ \Delta t.
$$

This can be interpreted as a stylized electricity cap: the contract pays whenever the spot price exceeds the strike $K$ during the delivery period.

The multi-period payoff is more natural for electricity-market applications than a payoff depending only on $S_T$.

### Crude Monte Carlo baseline

Crude Monte Carlo simulation establishes that the chosen high-strike cap is a genuine rare-event problem.

For the current parameterization, only about **0.05%** of crude Monte Carlo paths generate a positive payoff.

A simulation with $N=1{,}000{,}000$ paths produces a cap-value estimate on the order of

$$
1.5\times 10^{-5},
$$

while the payoff distribution remains highly sparse.

This gives importance sampling a clear role: most computational effort under the original measure is spent simulating paths that contribute nothing to the estimator.

### Fixed importance sampling

A constant Brownian control $\theta$ is introduced by simulating

$$
dX_t^\theta = \left[ \kappa \bigl( \mu(t) - X_t^\theta \bigr) + \sigma\theta \right]dt + \sigma  W_t.
$$

The corresponding Girsanov likelihood ratio is

$$
L_T^\theta = \exp\left( -\theta W_T -\frac{1}{2}\theta^2T \right). 
$$

The importance-sampling estimator is therefore based on the weighted payoff $Y^\theta L_T^\theta$.

A sweep over $\theta$ demonstrates the main trade-off:

- larger positive controls make high-price paths dramatically more common;
- estimator variance initially falls;
- overly aggressive tilting can eventually increase weight variability.

This provides a practical first indication of where the variance-minimizing control lies.

### Adaptive importance sampling

The constant control is then selected automatically by minimizing the empirical second moment

$$
J(\theta) = \mathbb{E}_{Q_\theta}\left[\left(Y^\theta L_T^\theta\right)^2 \right].
$$

Because the estimator is unbiased, minimizing $J(\theta)$ is equivalent to minimizing estimator variance within the chosen constant-control family.

The adaptive procedure follows the same structure developed in the original EPFL project:

1. coarse parameter search;
2. local refinement;
3. increasing Monte Carlo sample size;
4. convergence monitoring.

Common random numbers and nested Brownian samples are used to reduce noise in comparisons between candidate controls as the sample size grows.

Across independent random seeds, the selected control consistently settles around

$$
\theta^* \approx 11.7\text{--}11.8.
$$

Typical final results are:

- cap-value estimate around $1.54\times10^{-5}$;
- positive-payoff frequency around $49\%$ to $50\%$ under the tilted measure;
- variance reduction on the order of **several hundred times** relative to crude Monte Carlo.

The exact variance-reduction factor remains Monte Carlo dependent, so the project does not rely on a single seed-specific VRF. The robust conclusion is that adaptive importance sampling transforms an event occurring roughly once every 2,000 crude paths into an event occurring in about half of the tilted paths, while preserving a price estimate consistent with the crude Monte Carlo benchmark.

## EPFL stochastic-simulation foundation

The reusable numerical architecture originates from a rework of an EPFL MATH-414 Stochastic Simulation project.

Two derivative examples were used to develop and validate the methodology before applying it to power markets.

### European call

The first benchmark uses geometric Brownian motion,

$$
dS_t = rS_t dt + \sigma S_t dW_t,
$$

with payoff

$$
e^{-rT}(S_T-K)^+.
$$

The implementation progresses through crude Monte Carlo, fixed-drift importance sampling, adaptive importance sampling, and PDE-based optimal control. The Black-Scholes price provides an analytical benchmark.

### Up-and-out barrier call

The second benchmark uses the path-dependent payoff

$$
e^{-rT}(S_T-K)^+ \mathbf{1}\left( \max_{0\leq t\leq T}S_t \leq U \right).
$$

This example demonstrates why a constant drift can be structurally insufficient: simulations must be pushed upward toward the strike without simultaneously increasing barrier knock-outs.

A PDE-based state-dependent control resolves this by adapting the drift according to time and current state.

These two experiments now serve primarily as:

- the theoretical basis for the energy extension;
- validation of the importance-sampling machinery;
- the architecture from which the power-market models and experiments are built.

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

The separation is deliberate:

- `src/` contains reusable numerical methods and model components;
- `experiments/` contains parameter choices, numerical studies, diagnostics, and plots.

The same experiment structure is reused across the European call, barrier call, and power-cap extension.

## Running the experiments

The original option-pricing experiments can be run from the repository root with:

```bash
python3 -m experiments.run_all vanilla
```

or

```bash
python3 -m experiments.run_all barrier
```

or both:

```bash
python3 -m experiments.run_all all
```

The energy experiments currently live in the corresponding crude-MC, fixed-IS, and adaptive-IS experiment modules while the power-market extension is being developed.

## Current conclusions

The project has established three main results.

First, a seasonal mean-reverting electricity-price model combined with a high-strike multi-period cap payoff naturally creates a rare-event Monte Carlo problem.

Second, fixed importance sampling can substantially increase the frequency of high-price paths by shifting the Brownian measure.

Third, adaptive importance sampling can automatically identify a near-optimal constant control and reduce estimator variance by several hundred times while producing a cap-value estimate consistent with the crude Monte Carlo benchmark.

The energy extension therefore demonstrates that the importance-sampling ideas developed in the original financial-derivatives setting transfer naturally to a power-market problem with different dynamics and a multi-period delivery payoff.

## Next step: price spikes and Nordic calibration

The next stage is to introduce electricity-specific price spikes.

A natural extension is a mean-reverting jump model such as

$$
dX_t = \kappa\bigl(\mu(t)-X_t\bigr)dt + \sigma dW_t + dJ_t,
$$

or a separate fast mean-reverting jump factor.

This creates a more distinctly energy-market rare-event problem: high cap payoffs can arise not only from large diffusive moves, but also from explicit spike events.

The main numerical question will then become whether importance sampling should tilt:

- the Brownian component;
- the jump intensity or jump-size distribution;
- or both.

A later stage can replace the stylized parameters and sinusoidal seasonality with estimates based on Nord Pool data, making the model explicitly Nordic rather than only Nordic-power-inspired.

## Origin

The numerical foundation of this repository comes from a project completed in **EPFL MATH-414 Stochastic Simulation**.

The code has since been rewritten into a modular Python project and extended beyond the original assignment toward commodity and electricity-market applications.

## Keywords

**Keywords:** energy derivatives, electricity markets, power markets, Nordic power, Monte Carlo, importance sampling, adaptive importance sampling, rare-event simulation, variance reduction, mean reversion, seasonality, power caps, stochastic simulation, Girsanov theorem, optimal control, barrier options, quantitative finance
