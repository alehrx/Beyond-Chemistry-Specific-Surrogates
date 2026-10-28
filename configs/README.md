# Study configurations

One JSON file per training study, run with

    python scripts/run_training.py configs/<study>.json             # run, or resume after an interruption
    python scripts/run_training.py configs/<study>.json --dry-run   # print the plan only
    python scripts/run_training.py configs/<study>.json --stages post            # figures only
    python scripts/run_training.py configs/<study>.json --stages score post      # re-score + figures

Results: `runs/studies/<name>/` (config copy, metadata, `study.log`, `results.csv`, `figures/`);
networks: `runs/v2_nondim/<name>_s<seed>` and `..._lbfgs`. A study name is never reused with a
different configuration: change `name` for every new study. `mionet_77k.json` (the reported network,
seeds 42-45, hyperparameters from `runs/hpo_study`) is a complete example.

## Keys

**Top level**
- `name`: study and run-name prefix (`<name>_s<seed>`).
- `description`: free text, copied to the study metadata.
- `seeds`: list, e.g. `[42]` or `[42, 43, 44, 45]`.

**`hyperparameters`**: architecture, lr and loss weights. Three modes:
- `{"mode": "file", "path": "runs/hpo_study"}`: a folder holding `model_config.json` (any earlier HPO).
- `{"mode": "manual", "hidden_dim": 64, "num_hidden_layers": 7, "activation": "silu", "lr": 3.57e-4,
   "w_pde": 249, "w_ic": 1666, "w_bc": 299, "w_mc": 11.2}`: values set by hand.
- `{"mode": "run", "hpo_name": "hpo_rollout_new", "args": {...}}`: a new roll-out HPO, then its best
  trial. `args` are the options of `scripts/hpo_rollout.py` without the dashes, e.g.
  `{"steps": 10, "epochs": 250, "hidden": [16, 32, 64, 128], "layers": [2, 8], "lr_range": [1e-4, 3e-3],
   "pilot_mult": 2, "batches": 16, "no_project": true, "no_hard_bc": true, "with_dd": true}`.

**`dataset`**: input functions (`n_*`) and collocation points per function (`ny_*`):
`n_pde`, `ny_pde`, `n_ic`, `ny_ic`, `n_bc`, `ny_bc`, and `resample_every` (redraw the input
functions every N epochs). The validation set is fixed (150 x 1000 / 300 x 100 / 300 x 100).

**`training`** (Adam stage)
- `hard_bc`: add -Phi r^2/2 to the network output.
- `num_batches`: mini-batches per epoch, or `"auto"`: the fewest that fit in 70 % of the free GPU memory,
  measured with two probe steps at the start of the run (the choice is in the run's `training.log`,
  line `batch autotune`). `num_batches_min` (default 1) is the floor used with `"auto"`. Fewer batches
  = fewer optimiser steps per epoch, so the lr and the epoch counts may need adjusting.
- `epochs`: maximum epochs (a large number = no limit, early stopping ends the stage).
- `lr_decay`, `lr_decay_epochs`, `lr_end_factor`: linear decay of the lr to `lr_end_factor` times
  its initial value over `lr_decay_epochs` (null = over `epochs`), frozen afterwards.
- `patience`, `es_from_epoch`: early stopping after `patience` epochs without improvement, checked
  from epoch `es_from_epoch` on (0 = always).
- `monitor`: `"pi_val_loss"`, the held-out physics-informed validation loss (the only option);
  `monitor2`: a second metric to keep a checkpoint for in `best_pi/` (null = none).

**`lbfgs`**: `enabled`, `epochs`, `patience`, `lr`, `pool`, `k`, `c` (residual-based sampling),
`ny_pde` (points per function in the full batch), `monitor`, and optionally `num_batches`: the
batch count used while the adaptive set is drawn (default: the Adam stage's value). `"auto"` empties
the GPU cache after the draw; with a fixed count the full-batch step can run out of memory, as it
did for seed 43 of `mionet_77k` (originally study `wk25rcv`) with 32.

**`scoring`**: `enabled`, `tau_target` (window rule), `models`: `"adam"`, `"lbfgs"` or `"both"`,
`direct` (default true): the network is evaluated at the 51 scoring radii themselves; false = its
sensor values are interpolated onto them (the scores of studies before 2026-10-22). `fv_grid`
(default 51): the reference is solved with this many radial finite volumes and the errors are
computed at its own nodes, the cell centres plus the surface, so nothing is interpolated; null =
an evenly spaced 51-point grid with the reference interpolated onto it.

**`postprocess`**
- `training_curves`: training/validation loss figure per seed (`figures/<name>_s<seed>/`).
- `hpo`: search history and fANOVA importance (when the hyperparameters come from an HPO).
- `error_bins`: `{"enabled": true, "models": "lbfgs" | "adam", "ylim": null | value}`, the roll-out
  error over the seeds (bars = mean, whiskers = range); `script` overrides the path of the plotting
  script (default `scripts/figures/fig_error_bins.py`). `phi_error` (default true): the error
  against the peak dimensionless flux, over the same seeds.
