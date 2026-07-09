# Week 1 — PDEs, Neural Network Refresher & Automatic Differentiation

**Notebook:** `Week1.ipynb`

## Goal

Understand what a PDE is, refresh neural network intuition, and learn to differentiate a
neural network with respect to its **inputs** (not its weights) — the core operation that
makes PINNs possible.

## What This Notebook Does

Implements a 3-layer MLP from scratch in PyTorch and uses `torch.autograd.grad` to compute
first and second derivatives of the network output with respect to its input. The derivatives
are verified against the analytical solution `u(x) = sin(x)`, where `du/dx = cos(x)` and
`d²u/dx² = −sin(x)`.

## Tasks

1. **Build an MLP** — 3-layer network (1 → 32 → 32 → 1) with `tanh` activations, **1,153 parameters**
2. **Compute derivatives** — `du/dx` and `d²u/dx²` via `torch.autograd.grad` with `create_graph=True`
3. **Verify** — evaluate at 100 random points in `[−π, π]`
4. **Plot** — 3-subplot comparison of autograd vs analytical for `u`, `du/dx`, `d²u/dx²`

## Results

**Untrained network** (verifies autograd mechanics on a random function):

| Quantity | Max absolute error |
|---|---|
| \|u − sin(x)\| | 1.4429 |
| \|u′ − cos(x)\| | 1.3690 |
| \|u″ + sin(x)\| | 1.3005 |

Large errors are expected here — the point is that autograd correctly differentiates *whatever*
function the untrained network represents, not that the network already equals `sin(x)`.

**After training** on `sin(x)` for 3,000 epochs (bonus section):

| Quantity | Max absolute error |
|---|---|
| \|u − sin(x)\| | 0.005618 |
| \|u′ − cos(x)\| | 0.041079 |
| \|u″ + sin(x)\| | 0.112280 |

Once the network learns `sin(x)`, its autograd derivatives closely match `cos(x)` and `−sin(x)`.
Final training MSE reached `3e-6`.

## Key Concept

Differentiating the network **output** with respect to its **input** — not the weights — is the
one idea behind every PINN. `create_graph=True` is essential: it keeps the computation graph of
the first derivative alive so a second derivative can be taken.

## How to Run

```bash
pip install torch numpy matplotlib
jupyter notebook Week1.ipynb
```

Runs on CPU in seconds — no GPU needed.
