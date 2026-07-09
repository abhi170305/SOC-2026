# Week 3 — Canonical PDEs: Heat Equation & Wave Equation

**Notebook:** `Week3.ipynb`

## Goal

Scale up from ODEs to PDEs. Solve the heat and wave equations using the **DeepXDE** library,
and study how the number of collocation points affects accuracy.

## Problems

**Heat equation (parabolic):**
```
uₜ = 0.4 u_xx,   x ∈ [0,1],  t ∈ [0,1]
u(0,t) = u(1,t) = 0,   u(x,0) = sin(πx)
Analytical: u(x,t) = exp(−0.4π²t)·sin(πx)
```

**Wave equation (hyperbolic):**
```
uₜₜ = u_xx,   x ∈ [0,1],  t ∈ [0,1],  c = 1
u(x,0) = sin(πx),  uₜ(x,0) = 0,  u(0,t) = u(1,t) = 0
```

## Tasks & Results

### Part A — Heat Equation
Solved with a 4-layer, 64-neuron FNN (DeepXDE), 2000 domain + 200 boundary + 200 initial points,
10,000 Adam iterations.

**Final L² relative error: 3.14e-03** — excellent agreement with the analytical solution.
Solution surface, analytical surface, and absolute error are plotted as heatmaps.

### Part B — Collocation Sampling Experiment
Repeated Part A across collocation counts `[500, 1000, 2500, 5000, 10000]`:

| Collocation points | L² Relative Error |
|---|---|
| 500 | 7.46e-03 |
| 1000 | 1.36e-02 |
| … | (see notebook for full sweep) |

Accuracy generally improves with more points but with **diminishing returns** — beyond ~5000
points the bottleneck shifts to network capacity and training epochs rather than sampling density.

### Part C — Wave Equation
Solved with `uₜ(x,0) = 0` enforced via an `OperatorBC`, 5000 domain points, 15,000 iterations
(training took ~136 s).

**Final L² relative error: 9.21e-02** — noticeably higher than the heat equation. The wave
equation's oscillatory (periodic) time behaviour is harder for the PINN than the heat equation's
smooth exponential decay, echoing the spectral-bias difficulty from Week 2.

## Key Concept

DeepXDE separates *what the problem is* (geometry, PDE, BCs, ICs) from *how to solve it*
(network, optimiser). The PINN solution is a **continuous function** queryable at any `(x, t)` —
not a discrete grid like classical solvers produce.

**Heat vs Wave:** the heat solution decays smoothly (easy to fit); the wave solution oscillates
in time (harder, higher error). `tanh` is required over `relu` because `relu`'s second derivative
is zero almost everywhere, making the PDE residual gradient vanish.

## How to Run

```bash
pip install deepxde torch numpy matplotlib
jupyter notebook Week3.ipynb
```

DeepXDE auto-detects GPU. Total runtime is a few minutes across all three parts.
