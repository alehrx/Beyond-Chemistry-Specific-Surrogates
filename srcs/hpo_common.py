"""
hpo_common.py — shared infrastructure for hpo_kan.py / hpo_mionet.py.

Both scripts train with pure PI losses (PDE + IC + BC) but evaluate each
trial with the weight-independent DD MSE against a held-out PyBaMM simulation.
This separation is critical: w_pde, w_ic, w_bc are themselves hyperparameters
under optimisation, so a weighted PI loss cannot serve as a fair cross-trial
objective.

  - get_pi_dataset()   : builds (or loads from a disk cache) the PI training +
                         validation collocation tensors. Cached so that (a)
                         repeated HPO runs don't re-pay the GRF-profile
                         generation cost, and (b) hpo_kan.py and hpo_mionet.py
                         search on the *exact same* data when their data-
                         defining config matches — required for a fair
                         MLP vs KAN comparison.
  - get_dd_dataset()   : runs (or loads from disk cache) a single PyBaMM
                         simulation to produce the ground-truth DD validation
                         tensor X_val_dd used as the Optuna objective.
  - dd_val_loss()      : unweighted MSE(y_true, y_pred) on X_val_dd — the
                         weight-independent HPO objective.
  - hpo_train()        : Adam-only mini-training loop with Optuna pruning,
                         now reporting dd_val_loss at each prune checkpoint.
  - ACTIVATIONS        : C²-differentiable activation catalogue (the PI loss
                         differentiates the trunk twice via autograd, so
                         ReLU-family activations are excluded).
  - make_hpo_dir()      : auto-incrementing runs/<name> directory.
  - save_optuna_plots() : optional Optuna visualisation HTML (needs plotly).
"""

import os
import json
import hashlib
import logging

import numpy as np
import torch
import torch.nn as nn
import optuna
from pathlib import Path

from srcs.prepData import Config, data

# Only C² activations (PI loss takes 2nd derivatives of the trunk w.r.t. ξ).
ACTIVATIONS: dict = {
    "Tanh": nn.Tanh, "SiLU": nn.SiLU, "Softplus": nn.Softplus, "GELU": nn.GELU,
}

# Config fields that affect PI collocation data (architecture/lr/weights excluded).
_PI_DATA_FIELDS = ["electrode", "param_set", "b1_dim", "b2_dim", "Ny_pde", "Ny_bc",
                   "Ny_ic", "N_pde", "N_bc", "N_ic", "scale_length1", "scale_length2",
                   "scale_factor1", "scale_factor2", "t_tr",
                   # Everything below also changes the generated samples. They were missing, so a
                   # study that varied any of them silently reused the previous study's cached
                   # tensors - two radial meshes, or a 'grf' and a 'grf+parabola' profile sampler,
                   # collided on the same key.
                   "profile_mode", "phi_dist", "phi_span",
                   "xi_map", "xi_warp_p", "xi_colloc", "xi_quad", "profile_space", "n_fine",
                   # geometry resolves all four of the above, so it MUST be part of the key: two
                   # geometries leave xi_map/xi_colloc at the same 'auto' string and would collide.
                   "geometry", "geometry_law", "t_hi", "profile_amp_max",
                   "profile_layer_frac", "profile_layer_delta"]

# Config fields that affect the PyBaMM DD simulation output.
_DD_DATA_FIELDS = ["electrode", "param_set", "b1_dim", "b2_dim", "t_tr"]


def _cache_key(data_cfg: dict, val_sizes: dict, seed: int) -> str:
    payload = {k: data_cfg.get(k) for k in _PI_DATA_FIELDS}
    payload["val"] = val_sizes
    payload["seed"] = seed
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _dd_cache_key(data_cfg: dict, seed: int) -> str:
    payload = {k: data_cfg.get(k) for k in _DD_DATA_FIELDS}
    payload["seed"] = seed
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def get_pi_dataset(data_cfg: dict, val_sizes: dict, cache_dir: str, seed: int = 42):
    """Return ((X_pde, X_ic, X_bc), (X_val_pde, X_val_ic, X_val_bc)).

    `val_sizes` is a dict with keys N_pde/Ny_pde/N_ic/Ny_ic/N_bc/Ny_bc for the
    held-out validation set. Cached to `cache_dir` keyed by every field that
    affects the generated data (NOT hidden_dim/num_hidden_layers/lr/etc, so
    the cache is shared across every architecture trial and across hpo_kan.py
    / hpo_mionet.py whenever their data-defining config matches).
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(data_cfg, val_sizes, seed)
    cache_path = os.path.join(cache_dir, f"pi_dataset_{key}.pt")

    if os.path.exists(cache_path):
        blob = torch.load(cache_path, weights_only=False)
        logging.info(f"PI dataset loaded from cache: {cache_path}")
        return blob["train"], blob["val"]

    logging.info(f"PI dataset cache miss — generating (seed={seed}) …")
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        ds = data(Config(**data_cfg))
        train = (ds.data_pde(), ds.data_ic(), ds.data_bc())
        val = (ds.data_pde(N_pde=val_sizes["N_pde"], Ny_pde=val_sizes["Ny_pde"]),
               ds.data_ic(N_ic=val_sizes["N_ic"], Ny_ic=val_sizes["Ny_ic"]),
               ds.data_bc(N_bc=val_sizes["N_bc"], Ny_bc=val_sizes["Ny_bc"]))
    finally:
        np.random.set_state(rng_state)

    torch.save({"train": train, "val": val}, cache_path)
    logging.info(f"PI dataset cached to: {cache_path}")
    return train, val


def get_dd_dataset(data_cfg: dict, cache_dir: str, seed: int = 42):
    """Return X_val_dd: a PyBaMM-simulated ground-truth tensor for DD validation.

    Shape (1, res*n_time, 2+b1+b2+1) — columns: t, ksi, cs_0[b1], i[b2], y_true.
    Cached to `cache_dir` keyed on the fields that affect the simulation output
    (electrode, param_set, b1_dim, b2_dim, t_tr, seed).  The PyBaMM simulation
    is expensive (~minutes) but runs at most once per configuration.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = _dd_cache_key(data_cfg, seed)
    cache_path = os.path.join(cache_dir, f"dd_dataset_{key}.pt")

    if os.path.exists(cache_path):
        X_val_dd = torch.load(cache_path, weights_only=False)
        logging.info(f"DD dataset loaded from cache: {cache_path}")
        return X_val_dd

    logging.info(f"DD dataset cache miss — running PyBaMM simulation (seed={seed}) …")
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        ds = data(Config(**data_cfg))
        X_val_dd = ds.data_val()
    finally:
        np.random.set_state(rng_state)

    torch.save(X_val_dd, cache_path)
    logging.info(f"DD dataset cached to: {cache_path}")
    return X_val_dd


def dd_val_loss(trainer, X_val_dd, device: str) -> float:
    """Unweighted DD MSE — the weight-independent HPO objective.

    Computes mean((y_true - y_pred)²) on the held-out PyBaMM simulation tensor.
    Unlike pi_val_loss, this is independent of w_pde / w_ic / w_bc, so cross-
    trial comparisons remain valid when those weights are hyperparameters.
    """
    with torch.no_grad():
        return trainer.loss_dd(X_val_dd.to(device)).item()


def hpo_train(model, trainer, lr, loader_pde, loader_ic, loader_bc,
              X_val_dd, trial, epochs, prune_interval,
              lr_end_factor, device) -> float:
    """Adam-only mini-training loop with PI-only losses; DD MSE as Optuna objective.

    Training uses the PI loss (PDE + IC + BC) weighted by trainer.W_pde/ic/bc.
    Validation and pruning use dd_val_loss(X_val_dd) — weight-independent — so
    Optuna can meaningfully compare trials that explore different weight scales.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=lr_end_factor, total_iters=epochs)

    model.train()
    best_val = float("inf")
    for epoch in range(epochs):
        for batch_pde, batch_ic, batch_bc in zip(loader_pde, loader_ic, loader_bc):
            X_pde = batch_pde[0].to(device)
            X_ic  = batch_ic[0].to(device)
            X_bc  = batch_bc[0].to(device)

            def closure():
                optimizer.zero_grad()
                loss = (trainer.loss_pde(X_pde) * trainer.W_pde
                        + trainer.loss_ic(X_ic)  * trainer.W_ic
                        + trainer.loss_bc1(X_bc) * trainer.W_bc
                        + trainer.loss_bc2(X_bc) * trainer.W_bc)
                # mass-conservation term (searchable via w_cons); guarded so the
                # extra forward passes only run when the weight is active
                if getattr(trainer, "W_cons", 0.0) > 0:
                    loss = loss + trainer.loss_cons(X_pde) * trainer.W_cons
                loss.backward()
                return loss

            optimizer.step(closure)
        scheduler.step()

        if epoch % prune_interval == 0:
            model.eval()
            val = dd_val_loss(trainer, X_val_dd, device)
            model.train()
            if np.isnan(val) or np.isinf(val):
                raise optuna.TrialPruned()
            best_val = min(best_val, val)
            trial.report(val, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    model.eval()
    final_val = dd_val_loss(trainer, X_val_dd, device)
    return min(best_val, final_val)


def make_hpo_dir(name: str, base: str = "runs") -> Path:
    """Auto-incrementing runs/<name>, runs/<name>_001, … (never overwrites)."""
    root = Path(base) / name
    if not root.exists():
        root.mkdir(parents=True)
        return root
    i = 1
    while (cand := Path(base) / f"{name}_{i:03d}").exists():
        i += 1
    cand.mkdir(parents=True)
    return cand


def save_optuna_plots(study: optuna.Study, out_dir: Path) -> None:
    try:
        from optuna.visualization import (plot_optimization_history,
                                          plot_param_importances,
                                          plot_parallel_coordinate)
        plot_optimization_history(study).write_html(str(out_dir / "opt_history.html"))
        plot_param_importances(study).write_html(str(out_dir / "param_importance.html"))
        plot_parallel_coordinate(study).write_html(str(out_dir / "parallel_coord.html"))
    except Exception as exc:
        logging.warning(f"Could not generate Optuna plots (install plotly): {exc}")


# ---- optional PyBaMM one-step diagnostic of the HPO (hpo_rollout.py --with-dd) ----
# Moved here unchanged from train_nd.py, which no longer uses any labelled data. Five cases, one
# chemistry each, so no single cell dominates; both electrodes; isothermal SPM, cached in CACHE_DIR.
DD_PERIOD_S = 60.0
DD_SPEC = [
    dict(pm='Ai2020',      T=25.0, cr=1.0),
    dict(pm='Mohtat2020',  T=25.0, cr=2.0),
    dict(pm='Marquis2019', T=0.0,  cr=1.0),
    dict(pm='Chen2020',    T=0.0,  cr=2.0),
    dict(pm='Prada2013',   T=50.0, cr=2.0),
]


def build_val_reference(T, n_sensors=None, tau_max=None, with_groups=False, sensors=None, grid=None):
    """PyBaMM one-step reference over DD_SPEC (T: the train_nd module, for its grid settings).
    with_groups: also return the (S,) group label 10*case + electrode."""
    import torch
    from srcs_EM.reference import build_dd_reference
    n_s = n_sensors or T.N_SENSORS
    sensors = T.sensor_grid(n_s) if sensors is None else sensors
    Xs, Gs = [], []
    for i, sp in enumerate(DD_SPEC):
        X, G = build_dd_reference(T.CACHE_DIR, sp['pm'], [float(sp['T'])], float(sp['cr']),
                                  DD_PERIOD_S, n_s, tau_max or T.TAU_MAX, return_groups=True,
                                  sensors=sensors, grid=grid or T.REF_GRID, r_pts=T.REF_R_PTS)
        if X.shape[0] == 0:
            continue
        Xs.append(X); Gs.append(G + 10 * i)
    if not Xs:
        raise RuntimeError('DD_SPEC produced no reference samples at all')
    return (torch.cat(Xs), torch.cat(Gs)) if with_groups else torch.cat(Xs)
