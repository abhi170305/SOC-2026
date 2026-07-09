# Weeks 7–8 — Track D: Neural Operators (Fourier Neural Operator)

**Notebooks:**
- `week7_progress.ipynb` — Week 7 checkpoint (FNO on fixed-ν Burgers)
- `final_notebook_colab.ipynb` — Week 8 final project (parametric FNO + all experiments)

**Also submitted:** `slides.pdf` / `slides.pptx` (10-slide deck), `README.md` (project description)

## Goal

Learn the **solution operator**, not just one solution. Instead of training a new network for
every PDE instance (as PINNs do in Weeks 1–5), train a single Fourier Neural Operator once that
maps any initial condition to its solution:

```
G: u₀(x) → u(x, T=1)
```

## The Paradigm Shift

| | PINN (Weeks 1–5) | Neural Operator (this project) |
|---|---|---|
| Learns | One solution for one IC | The solution map for all ICs |
| Input → Output | `(x,t) → u` | `u₀(x) → u(x,T)` |
| New IC | Retrain from scratch | Instant inference |

## Problem

1D Burgers' equation, `ν = 0.01/π` (Week 7) and `ν ∈ [0.01, 0.1]` sampled per-trajectory (Week 8).
Training data: initial conditions sampled from Gaussian random fields, each solved with a
**pseudo-spectral RK4 + integrating factor** solver (no `solve_ivp` — avoids stiffness and
complex-state failures), with 2/3 dealiasing.

---

## Week 7 — Checkpoint (`week7_progress.ipynb`)

FNO implemented **from scratch** in PyTorch (spectral convolution + local convolution layers),
trained on 1,000 GRF trajectories at fixed ν.

- Dataset: 1,000 train + 200 test trajectories, **0 NaN**
- Model: **549,569 parameters** (16 Fourier modes, 64 hidden channels, 4 layers)
- **Final test L² relative error: 4.54%** — target < 5% **PASS**

Test error dropped from ~8.4% early to 4.54% over 500 epochs.

---

## Week 8 — Final Project (`final_notebook_colab.ipynb`)

Parametric FNO conditioned on viscosity ν (3 input channels: `u₀`, `x`, `ν`), trained on
Colab **Tesla T4 GPU** (CUDA 12.8). Dataset generation took 300.3 s.

- Model: **549,633 parameters**
- **Final test L² relative error: 4.70%** (in-range ν) — target < 5% **PASS**

### Generalisation experiment

| ν | In training range? | L² Error |
|---|---|---|
| 0.001 | ✗ extrapolation | **0.3576 (35.76%)** |
| 0.05 | ✓ in range | 0.0375 (3.75%) |
| 0.1 | ✓ edge of range | 0.0561 (5.61%) |

The FNO **interpolates well within its training ν range** but error jumps ~10× under
extrapolation to ν=0.001 — it is a learned interpolator, not a physics solver.

### Data efficiency

| Training trajectories | Test L² Error |
|---|---|
| 50 | 0.5950 (59.5%) |
| 100 | 0.1377 (13.8%) |
| 250 | 0.0578 (5.78%) |
| 500 | 0.0536 (5.36%) |
| 1000 | 0.0578 (5.78%) |

An **accuracy cliff**: 50 trajectories are unusable (59.5%), but error collapses to ~5.8% by 250
trajectories, then plateaus — operator learning needs a critical mass of data, then saturates.

### Inference time — FNO vs 100 independent PINNs

| Method | Time |
|---|---|
| FNO (100 ICs, one batched pass) | **0.0065 sec** |
| 100 × PINN (independent trainings) | 771.4 sec (12.9 min) |
| **Speedup** | **≈ 118,170×** |

Once trained, a single FNO forward pass solves all 100 instances ~118,000× faster than training
100 separate PINNs.

## Key Concept — When to Use Which

**Use a neural operator when:** the same PDE family is solved repeatedly across many conditions;
inference speed matters; you can afford upfront training-data generation.

**Use a per-instance PINN when:** you need one specific solve; you're recovering an unknown
parameter (inverse problems, Week 6); generating operator training data is too expensive.

Neural operators and PINNs are **complementary tools, not competitors**. Train once, solve
forever — *within the training distribution*.

## How to Run

**Colab (recommended for Week 8):**
1. Upload `final_notebook_colab.ipynb`
2. Runtime → Change runtime type → **T4 GPU** → Save
3. Runtime → Run all

```bash
# Local
pip install torch numpy matplotlib scipy pandas
```

Data generation ~2–3 min (CPU-bound), FNO training ~3–5 min on GPU.

## Submission Contents

```
submission/
├── week7_progress.ipynb        # Week 7 checkpoint
├── final_notebook.ipynb        # local version (CUDA required)
├── final_notebook_colab.ipynb  # Colab-ready version
├── slides.pdf / slides.pptx    # 10-slide presentation
└── README.md                   # project description
```
