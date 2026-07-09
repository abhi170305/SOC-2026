# Week 5 — Training Pathologies: Why PINNs Fail & How to Fix Them

**Notebook:** `Week5_TrainingPathologies.ipynb`

## Goal

Understand the three main reasons PINNs fail to train, and implement two fixes — **Fourier
feature embedding** and **gradient-norm loss balancing** — on the Week 4 Burgers' PINN.

## The Three Failure Modes

1. **Spectral bias** — MLPs learn low frequencies first; high-frequency solutions (like Burgers'
   shock) converge slowly or not at all
2. **Loss imbalance** — PDE, IC, and BC losses have competing gradient magnitudes; one dominates
3. **Causal violation** — the network tries to satisfy `t = 1` before `t = 0` is learned (read
   only, not implemented)

## Tasks

1. **Fourier feature embedding** — map `(x,t)` to `[sin(Bv), cos(Bv)]` with `B ~ N(0, σ²)`,
   `d = 128` (256-dim embedding), tested at σ = 1 and σ = 10
2. **Gradient-norm loss balancing** — rescale `λ_bc`, `λ_ic` each step from gradient norms
3. **Comparative experiment** — vanilla vs Fourier vs gradient-norm on identical Burgers' setup
4. **Analysis** — which fix helped, and which loss term dominates

## Results

All variants trained for 10,000 epochs on the same Burgers' setup:

| Variant | Final Loss | L² Relative Error | Improvement vs Vanilla |
|---|---|---|---|
| 1. Vanilla PINN | 0.1825 | 0.3068 | — |
| 2. Fourier Features (σ=10) | 0.00052 | 0.3013 | +1.8% |
| 2b. Fourier Features (σ=1) | 0.0511 | **0.1052** | **+65.7%** |
| 3. Gradient-Norm Balancing | 0.0492 | 0.4341 | −41.5% |

**Key finding:** In this run, **Fourier features with σ=1 helped the most** (66% error reduction),
while σ=10 drove the training loss far down (0.00052) but did **not** improve the actual solution
accuracy — a sign that σ=10 was too high, injecting frequencies faster than the true solution
requires. Gradient-norm balancing actually *worsened* the L² error here, showing that loss-term
rebalancing is not automatically beneficial when the dominant challenge is spectral (shock
resolution) rather than loss imbalance.

### Gradient-norm diagnostic (vanilla PINN, final state)

| Loss term | Gradient norm |
|---|---|
| ‖∇L_pde‖ | **3.8482** (largest) |
| ‖∇L_bc‖ | 0.0105 |
| ‖∇L_ic‖ | 0.3590 |

The **PDE residual gradient dominates** by ~10× over the IC and ~370× over the BC — the optimiser
spends most of its effort on the PDE term while the boundary/initial conditions are comparatively
neglected, explaining why manual loss weighting was needed in Week 4.

## Key Concept

The PINN loss is a multi-task learning problem with competing gradients. The σ parameter in
Fourier embedding matters critically: too small barely helps, too large injects excessive
high frequencies and hurts convergence. The right value roughly matches the highest frequency
present in the true solution.

## How to Run

```bash
pip install torch numpy matplotlib scipy pandas
jupyter notebook Week5_TrainingPathologies.ipynb
```

GPU recommended (uses CUDA). Four 10,000-epoch trainings take several minutes on GPU.
