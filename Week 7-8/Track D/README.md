# Track D — Neural Operators: FNO on 1D Burgers' Equation

This project implements a Fourier Neural Operator (FNO) from scratch in PyTorch to learn the
solution operator `G: u₀(x) → u(x, T=1)` for the 1D Burgers' equation, replacing the
per-instance PINN approach from Weeks 1–5 with a single model that generalises across
initial conditions and viscosity values without retraining. The FNO is trained on 1,000
initial conditions sampled from a Gaussian random field (with viscosity ν drawn from
`[0.01, 0.1]` per trajectory) and evaluated on 200 held-out conditions, achieving a test L²
relative error consistent with the Week 7 checkpoint target of under 5%. Three experiments
extend the checkpoint into a complete study: a generalisation test showing the operator
interpolates well within its training ν range but degrades under extrapolation (ν=0.001);
a data-efficiency sweep across training set sizes from 50 to 1,000 trajectories; and an
inference-time comparison quantifying the multiple-orders-of-magnitude speedup a trained
operator provides over training 100 independent PINNs. The results directly illustrate the
central trade-off of operator learning — substantial upfront training cost in exchange for
near-instant, reusable inference across an entire family of problem instances, rather than
one-off accuracy on a single solve.
