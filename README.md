# A parameter-free physics-informed MIONet for particle diffusion

One trained network advances the lithium concentration profile inside an electrode particle of the
single particle model (SPM), for **either electrode of any chemistry, at any temperature**. It is
trained on the governing equation alone: no labelled simulation data enter the loss or the
selection of checkpoints, which uses a held-out physics-informed validation set.

## The parameter-free formulation

With `xi = r/R`, `tau = D_s(T) t / R^2`, `x = c/c_max` and the dimensionless surface flux `Phi`,
solid-phase diffusion becomes

```
dx/dtau = xi^-2 d/dxi ( xi^2 dx/dxi ),    -dx/dxi|_1 = Phi,    dx/dxi|_0 = 0
```

which contains no material, geometric or thermal parameter. The cell enters only through the two
scalars `(Phi, tau)`, computed in closed form by `srcs_EM/chemistry.py` from a PyBaMM parameter set,
the current, the step length and the temperature. The network maps `(x0, Phi, tau) -> x(tau, xi)`.

## Install

```bash
pip install -r requirements.txt
```

Run everything from the repository root.

## Use

```python
import numpy as np
from srcs_EM import Surrogate, Chemistry

sur  = Surrogate.from_run("runs/v2_nondim/trained_model_s42_lbfgs")  # the reported network (seed 42)
chem = Chemistry.from_pybamm("Chen2020", "Negative")         # any electrode of any parameter set

x = np.full(sur.n, 0.8)                                      # stoichiometry at the sensors
x = sur.step(x, I=-5.0, dt=20.0, chem=chem, T=298.15)        # one 20 s step at -5 A
```

`step` converts `(I, dt, T)` to `(Phi, tau)`, splits a step longer than the trained window into equal
sub-windows, and applies the lithium-balance projection after each one. Use `sur.sensors` for the
radial positions of the state.

Two deployment rules decide accuracy in use:

- **Step length.** The network is least accurate per call at very small `tau`, so it should not be
  called once per measurement sample. `Surrogate.plan_window` grows a window until `tau` reaches a
  target or the current changes, whichever comes first; the paper scores with `tau <= 0.05`.
- **Intermediate outputs are almost free.** `tau` is a network input, so `Surrogate.step_series`
  reports the profile at every sample time inside a window from one call, without feeding back.

## Layout

```
srcs_EM/        the operator (mionet_nd), the parameter-free transform (chemistry), deployment
                (surrogate), the exact series solution of the dimensionless particle (exact),
                PyBaMM one-step references for the HPO's optional diagnostic (reference), and the
                SPMe / multi-slice cell models (electrolyte, voltage, cell, multislice)
srcs/           sampler of input functions and collocation points (prepData), physics-informed
                training (train_nd, training, trainingTracker), L-BFGS refinement (refine_lbfgs),
                hyperparameter search (hpo_rollout, hpo_common), scoring against PyBaMM (evaluation)
scripts/
  run_training.py         one study end to end from a JSON config: HPO, Adam, L-BFGS, scoring, figures
  supervise_training.py   relaunches a study after a crash; interrupted runs resume exactly
  monitor_training.py     live training curves in TensorBoard
  cell_models/            the network inside SPM, SPMe and multi-slice models, against a PyBaMM DFN
  figures/                the paper's data figures (make_figures.py regenerates all of them)
configs/        study configurations (README.md lists every key)
runs/
  v2_nondim/      the four reported networks, trained_model_s42..s45, each with its Adam stage
                  (<name>) and its L-BFGS refinement (<name>_lbfgs)
  studies/        trained_model/: the study record of the four networks (config, metadata, and
                  results.csv, the scores the figures are drawn from)
  hpo_study/      the hyperparameter search behind the reported configuration (study.db)
figures/        the paper's data figures (PDF)
```

