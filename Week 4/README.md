# Week 4 — Burgers' Equation, Boundary Conditions & Classical Solver Comparison

**Notebook:** `Week4.ipynb`

## Goal

Reproduce the benchmark PINN result from Raissi et al. 2019 (Section 3.1) **from scratch in pure
PyTorch** (no DeepXDE), implement hard boundary conditions via trial functions, and develop a
clear view of where PINNs win and where classical solvers win.

## Problem — Burgers' Equation

```
uₜ + u·uₓ = (ν/π)·uₓₓ,   x ∈ [−1, 1],  t ∈ [0, 1]
u(−1,t) = 0,  u(1,t) = 0,  u(x,0) = −sin(πx),  ν = 0.01/π
```

Burgers' equation is the standard PINN benchmark: nonlinear, develops a near-shock at `t = 1`,
and has a known reference solution (computed here via a spectral method, shape 100 × 256).

## Setup

4-layer MLP, tanh activations, **20,601 parameters**, Xavier initialisation, 10,000 Adam epochs,
10,000 interior + 200 boundary + 100 initial collocation points.

## Tasks & Results

### Part A — Soft BCs (boundary loss penalty)
Boundary and initial conditions enforced as weighted loss terms.

**L² Relative Error: 2.44e-01**

Solution field, absolute error vs the spectral reference, and training loss curves are plotted.
Error concentrates near the shock region at `t = 1, x = 0`.

### Part B — Hard BCs (trial function)
Boundary conditions enforced *exactly* through the construction:

```
u(x,t) = (1 − x²)·t·NN(x,t) + (−sin(πx))·(1 − t)
```

This satisfies `u(±1, t) = 0` and `u(x, 0) = −sin(πx)` by construction, removing both the BC and
IC loss terms — only the PDE residual is minimised.

**Comparison:**

| Method | L² Relative Error |
|---|---|
| Soft BCs | **2.44e-01** |
| Hard BCs | 4.71e-01 |

In this run, **soft BCs achieved the lower error**. While hard BCs guarantee exact boundary
satisfaction and remove a competing loss term, the specific trial-function construction here
interacted with the strong nonlinearity and shock formation in a way that made optimisation
harder. On other problems (simpler geometry, no shock), hard BCs often win — the result is
problem-dependent.

### Part C — Written (PINNs vs FEM)
Based on Grossmann et al. 2023: PINNs rarely beat FEM for **forward** problems on smooth domains
(FEM hits 1e-6 error in milliseconds). PINNs win on **inverse problems, parameter identification,
high-dimensional PDEs, and data assimilation from sparse sensors** — cases where FEM is
inapplicable rather than merely effortful.

## Key Concept

Burgers' near-shock at `t = 1` is where the PINN has highest error. Classical solvers resolve
this perfectly with mesh refinement; PINNs achieve acceptable but not competitive accuracy in
this regime. PINNs and FEM are complementary tools, not competitors.

## How to Run

```bash
pip install torch numpy matplotlib scipy
jupyter notebook Week4.ipynb
```

Runs on CPU. Two 10,000-epoch trainings (soft + hard) take several minutes total.
