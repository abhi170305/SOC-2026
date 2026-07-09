# Week 2 — The PINN: Architecture, Loss Function & Harmonic Oscillator

**Notebook:** `Week2.ipynb`

## Goal

Build a complete, working PINN from scratch on the damped harmonic oscillator, and observe
spectral bias — the failure of neural networks to fit high-frequency solutions.

## Problem

The damped harmonic oscillator:

```
m ẍ + μ ẋ + kx = 0,   x(0) = 1,  ẋ(0) = 0
m = 1,  μ = 0.1,  ω₀ = √(k/m)
```

The PDE residual itself becomes the loss — there are no labelled data points. Initial conditions
are enforced as an additional weighted penalty term:

```
L_total = L_pde + 100 · L_ic
```

## Tasks

1. **Frequency sweep** — train the PINN for `ω₀ ∈ {1, 5, 10, 15, 20}`, recording final loss and L² error
2. **Written reflection** — explain spectral bias and propose fixes

## Results

| ω₀ | Final Loss | L² Error |
|---|---|---|
| 1 | 3.04e-03 | **0.0075** |
| 5 | 2.71e-02 | **0.0116** |
| 10 | 2.62e+00 | **0.0209** |
| 15 | 1.49e+01 | **0.2664** |
| 20 | 9.94e+01 | **0.9999** |

The PINN is accurate for low frequencies (under 3% error at ω₀ ≤ 10) but **fails completely at
ω₀ = 20**, where the L² error reaches 0.9999 — the network essentially cannot represent the
rapid oscillation at all. The worst case (ω₀ = 20) is plotted showing predicted vs exact `x(t)`.

## Key Concept — Spectral Bias

Neural networks trained by gradient descent learn low-frequency components first and
high-frequency components slowly or not at all. The frequency sweep demonstrates this directly:
as ω₀ increases, the target solution oscillates faster and the PINN's error grows sharply.

**Proposed fixes** (implemented in Week 5): Fourier feature embeddings, sinusoidal activations
(SIREN), adaptive collocation sampling, and curriculum learning on the time horizon.

## How to Run

```bash
pip install torch numpy matplotlib scipy
jupyter notebook Week2.ipynb
```

Runs on CPU. The full frequency sweep (5 × 10,000 epochs) takes a few minutes.
