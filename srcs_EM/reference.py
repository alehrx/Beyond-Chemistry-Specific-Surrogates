"""
srcs_EM.reference - PyBaMM one-step reference transitions for the HPO's optional diagnostic.

The one-step error of the net against PyBaMM SPM, in the dimensionless (Phi, tau) space, at several
temperatures and on both electrodes. Training never uses it (it is physics-informed only, with a
held-out physics-informed validation set); only srcs/hpo_rollout.py --with-dd records it.

Row layout (as srcs.training.Trainer.loss_dd): [t = tau/tau_max, xi, cs_0(n_sensors, pinned),
Phi, y_true(pinned)] -> tensor (S, n_sensors, 2 + n_sensors + 2). Cached on the definition.
"""
import os
import json
import hashlib
import logging

import numpy as np
import torch

from .chemistry import Chemistry

log = logging.getLogger(__name__)


def build_dd_reference(cache_dir, param_set='Mohtat2020', T_list=(-5.0, 25.0, 45.0), c_rate=2.0,
                       period_s=60.0, n_sensors=16, tau_max=0.2, thermal='isothermal', return_groups=False,
                       sensors=None, grid='legacy', r_pts=32):
    """One-step DD samples from PyBaMM SPM cycles (discharge / charge / CV hold) at each T in
    `T_list`, both electrodes, through `Chemistry.to_dimless`. Steps with tau > tau_max are
    dropped (the surrogate sub-steps those at deployment). Cached under `cache_dir`.
    `thermal`: 'isothermal' (cell T = ambient, one tau per (T, electrode)) or 'lumped' (PyBaMM
    lumped thermal model; the per-step X-averaged cell temperature feeds Chemistry.to_dimless,
    so the samples carry the self-heating T excursions - a denser (Phi, tau) coverage).

    `sensors`: the xi positions of the branch-1 inputs, (n_sensors,) increasing, ending at 1.
    Defaults to linspace(0, 1, n_sensors). Pass the network's own grid whenever it is not uniform.

    `grid`: how the PyBaMM radial solution is mapped onto those positions.
      'legacy' - solve with var_pts r_n = n_sensors and read row k as xi = k/(n_sensors-1).
                 PyBaMM's finite-volume concentration lives at CELL CENTRES, (k+0.5)/M, so at
                 M = 16 this labels the outermost node (true xi = 0.969) as the surface and
                 shifts every node inwards by up to half a cell. Kept as the default because it
                 is the mapping every existing cached reference and published number was built
                 with; it is also only usable on a uniform grid.
      'fv'     - solve with var_pts r_n = `r_pts`, take the true cell centres, anchor xi = 1 with
                 PyBaMM's own particle SURFACE concentration and xi = 0 with the first cell
                 (dc/dxi = 0 there), then interpolate onto `sensors`. Mesh-independent, and the
                 only correct option when the sensors cluster inside one PyBaMM cell.
    """
    T_list = [float(T) for T in T_list]
    if sensors is None:
        sensors = np.linspace(0, 1, n_sensors)
    sensors = np.asarray(sensors, dtype=float)
    if len(sensors) != n_sensors:
        raise ValueError(f"sensors has {len(sensors)} points, n_sensors is {n_sensors}")
    if grid not in ('legacy', 'fv'):
        raise ValueError("grid must be 'legacy' or 'fv'")
    if grid == 'legacy' and not np.allclose(sensors, np.linspace(0, 1, n_sensors)):
        raise ValueError("grid='legacy' only maps onto a uniform sensor grid; use grid='fv'")
    key_fields = [param_set, T_list, float(c_rate), float(period_s), int(n_sensors), float(tau_max)]
    if thermal != 'isothermal':
        key_fields.append(thermal)                   # isothermal keeps the pre-existing cache keys
    if grid != 'legacy':                             # ... and so does the legacy radial mapping
        key_fields += [grid, int(r_pts), [round(float(x), 9) for x in sensors]]
    key = hashlib.md5(json.dumps(key_fields).encode()).hexdigest()[:10]
    path = os.path.join(cache_dir, f'dd_nd_{key}.pt'); gpath = path.replace('.pt', '_groups.pt')
    if os.path.exists(path) and (not return_groups or os.path.exists(gpath)):
        log.info("DD reference (multi-T, both electrodes) loaded from cache: %s", path)
        X = torch.load(path, weights_only=False)
        return (X, torch.load(gpath, weights_only=False)) if return_groups else X
    import pybamm
    log.info("DD reference cache miss - running PyBaMM SPM %s (%s) at T=%s C, %.2fC, period %.0f s ...",
             param_set, thermal, T_list, c_rate, period_s)
    pybamm.set_logging_level('ERROR')
    xi = sensors; rows = []; groups = []                               # group = 2*T_index + electrode
    M = n_sensors if grid == 'legacy' else int(r_pts)                  # PyBaMM radial resolution
    chems = {el: Chemistry.from_pybamm(param_set, el) for el in ('Negative', 'Positive')}
    for iT, T_C in enumerate(T_list):
        p = pybamm.ParameterValues(param_set); T_k = 273.15 + T_C
        p['Initial temperature [K]'] = T_k; p['Ambient temperature [K]'] = T_k
        I_app = float(p['Nominal cell capacity [A.h]']) * c_rate; p.set_initial_stoichiometries(1)
        exp = pybamm.Experiment([f"Discharge at {I_app} A until 3.2 V", f"Charge at {I_app} A until 4.2 V",
                                 "Hold at 4.2 V until 0.05 C"], period=f"{period_s} seconds")
        opts = {} if thermal == 'isothermal' else {"thermal": thermal}
        sol = pybamm.Simulation(pybamm.lithium_ion.SPM(opts), parameter_values=p,
                                var_pts={"x_n": 10, "x_s": 10, "x_p": 10, "r_n": M, "r_p": M},
                                experiment=exp).solve()
        t = sol["Time [s]"].entries; I = sol["Current [A]"].entries
        T_cell = (np.full_like(t, T_k) if thermal == 'isothermal'
                  else sol["X-averaged cell temperature [K]"].entries)
        prof = {'Negative': np.mean(sol["Negative particle concentration"].entries, axis=1),
                'Positive': np.mean(sol["Positive particle concentration"].entries, axis=1)}   # (r, t)
        if grid == 'fv':
            # true FV cell centres, plus the two ends: xi=0 by the symmetry condition (dc/dxi=0,
            # so the first cell value carries over) and xi=1 from PyBaMM's surface concentration.
            nodes = (np.arange(M) + 0.5) / M
            xi_ref = np.concatenate([[0.0], nodes, [1.0]])
            prof = {el: np.vstack([p[:1], p,
                                   sol[f"X-averaged {el.lower()} particle surface concentration"].entries[None, :]])
                    for el, p in prof.items()}
            prof = {el: np.stack([np.interp(sensors, xi_ref, p[:, k]) for k in range(p.shape[1])], axis=1)
                    for el, p in prof.items()}                                         # -> (n_sensors, t)
        for iel, (el, ch) in enumerate(chems.items()):
            for k in range(len(t) - 1):
                dt = float(t[k + 1] - t[k]); phi, tau = ch.to_dimless(float(I[k]), dt, float(T_cell[k]))
                if tau > tau_max or dt <= 0:
                    continue                                              # one-step samples only
                c0 = prof[el][:, k]; c1 = prof[el][:, k + 1]; pin = c0[0]
                blk = np.zeros((n_sensors, 2 + n_sensors + 2))
                blk[:, 0] = tau / tau_max; blk[:, 1] = xi; blk[:, 2:2 + n_sensors] = c0 - pin
                blk[:, 2 + n_sensors] = phi; blk[:, -1] = c1 - pin
                rows.append(blk); groups.append(2 * iT + iel)
    X = torch.tensor(np.stack(rows), dtype=torch.float32); G = torch.tensor(groups, dtype=torch.long)
    os.makedirs(cache_dir, exist_ok=True); torch.save(X, path); torch.save(G, gpath)
    log.info("DD reference built: %d one-step samples x %d nodes -> %s", X.shape[0], X.shape[1], path)
    return (X, G) if return_groups else X


def stratified_mse(err_sq, groups):
    """err_sq: per-sample mean squared error (S,); groups: (S,) int labels. Returns (mean over groups of the
    group MSE, worst group MSE, {group: mse}). Every operating regime (T, C-rate, electrode) counts equally,
    instead of every sample: the pooled mean is dominated by the many low-|Phi| warm / CV-tail samples."""
    per = {}
    for g in torch.unique(groups):
        per[int(g)] = float(err_sq[groups == g].mean())
    vals = torch.tensor(list(per.values()))
    return float(vals.mean()), float(vals.max()), per


@torch.no_grad()
def dd_metrics(model, X, groups=None):
    """{'mse', 'rmse', 'surface_rmse', 'rel_to_step', 'n'} of `model` (MIONetND) on a reference
    tensor from `build_dd_reference`, in x = c/c_max units; with `groups` also 'mse_strat' (mean of the
    per-(T, C-rate, electrode) MSEs), 'mse_worst' (largest group MSE) and 'mse_groups'."""
    n_s = X.shape[1]
    t, xi, c0, phi, y = X[..., :1], X[..., 1:2], X[..., 2:2 + n_s], X[..., 2 + n_s:3 + n_s], X[..., -1:]
    e = model(t, xi, c0, phi) - y
    step = (y - c0).pow(2).mean().sqrt()
    out = dict(mse=float(e.pow(2).mean()), rmse=float(e.pow(2).mean().sqrt()),
               surface_rmse=float(e[:, -1].pow(2).mean().sqrt()),
               rel_to_step=float(e.pow(2).mean().sqrt() / step), n=int(X.shape[0]))
    if groups is not None:
        strat, worst, per = stratified_mse(e.pow(2).mean(dim=(1, 2)), groups.to(e.device))
        out.update(mse_strat=strat, mse_worst=worst, mse_groups={str(k): v for k, v in per.items()})
    return out
