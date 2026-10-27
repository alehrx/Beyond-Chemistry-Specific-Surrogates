"""
evaluation.py
=============
The scoring engine: every trained model x every cell x every protocol, scored on a common
radial grid, with the per-step cost measured on a single CPU core. Writes a tidy CSV that the
paper figures read, and caches every PyBaMM reference so adding a model never re-solves them.

    python srcs/evaluation.py                       # every model in runs/v2_nondim
    python srcs/evaluation.py --models A B          # only these run folders
    python srcs/evaluation.py --cells Chen2020      # only these cells
    python srcs/evaluation.py --force               # recompute rows already in the CSV
    python srcs/evaluation.py --no-timing           # skip the single-core cost measurement

It is called by scripts/run_training.py (the score stage); it is rarely run by hand.

Cells, named as in the paper (= the PyBaMM parameter set):
    Ai2020        pouch,      LiCoO2 / graphite
    Marquis2019   pouch,      LiCoO2 / graphite
    Mohtat2020    pouch,      NMC / graphite
    Chen2020      21700 cyl,  NMC811 / graphite-SiOx
    Prada2013     18650 cyl,  LiFePO4 / graphite

Protocols: CC discharge + CC-CV charge at 0.5C / 1C / 3C, and a 3C-scaled DST, each run
isothermally at 0, 25 and 50 C.

WHY THE METRICS ARE COMPUTED ON A DENSE GRID.  Each network carries the profile at its own 16
sensor positions, and those positions differ between the uniform and the surface-clustered mesh.
Scoring each network on its own nodes would compare two different quantities. Every roll-out is
therefore linearly interpolated from its sensor grid onto one common dense xi grid, and the
PyBaMM reference is brought to that same grid from the true finite-volume cell centres with the
particle surface concentration as the xi = 1 anchor. The volume-weighted column additionally
weights each dense node by the shell of the particle it stands for.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SM_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, SM_ROOT)

import pybamm                                              # noqa: E402
import torch                                               # noqa: E402
from srcs_EM import Surrogate, Chemistry                 # noqa: E402

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ definitions
CELLS = {                       # display name -> (PyBaMM parameter set, format)
    'Ai2020':      ('Ai2020',      'pouch'),
    'Marquis2019': ('Marquis2019', 'pouch'),
    'Mohtat2020':  ('Mohtat2020',  'pouch'),
    'Chen2020':    ('Chen2020',    '21700 cylindrical'),
    'Prada2013':   ('Prada2013',   '18650 cylindrical'),
    # Not part of the headline grid: the stress case for the constant-D_s assumption, run with
    # --cells ORegan. Its D_s varies 9.8x over stoichiometry on the cathode and 38x on the anode.
    # READ THE CATHODE ONLY. The anode's D_s dips to 2.9e-16 at mid-stoichiometry, which makes
    # j0_aux tiny and drives |Phi| to 5.3 / 10.9 / 32.6 at 0.5C / 1C / 3C - up to 11x outside the
    # trained |Phi| <= 3. Its huge errors are extrapolation in flux, NOT the D_s(x) assumption
    # failing, and it would fail there with a constant D_s too.
    'ORegan2022':  ('ORegan2022',  '21700 cylindrical, D_s(x)'),
}
HEADLINE_CELLS = ['Ai2020', 'Marquis2019', 'Mohtat2020', 'Chen2020', 'Prada2013']
# The names used until 2026-10-22. They still name the cached PyBaMM references and saved fields
# (so nothing is re-solved), and older results CSVs are read through them.
LEGACY = {'Ai2020': 'LCO 2.3 (Ai)', 'Marquis2019': 'LCO 0.6 (Marquis)', 'Mohtat2020': 'NMC PC (Mohtat)',
          'Chen2020': 'NMC Cyl (Chen)', 'Prada2013': 'LFP Cyl (Prada)', 'ORegan2022': 'NMC LGM50 (ORegan)'}
FROM_LEGACY = {v: k for k, v in LEGACY.items()}
PROTOCOLS = ['CC-CV 0.5C', 'CC-CV 1C', 'CC-CV 3C', 'DST 3C']
# Isothermal runs at each of these. 0 C rather than -5 C: at -5 C the Marquis anode reaches
# |Phi| = 2.93 at 3C, which sits on the edge of the trained envelope (|Phi| <= 3) and the error
# there reports extrapolation rather than the surrogate's accuracy. At 0 C it is 2.06.
T_LIST = [0.0, 25.0, 50.0]
T_C = 25.0                      # the reference temperature, used when a single value is needed
PERIOD_S = 20.0                 # CC-CV sampling period, as the paper's roll-outs
CV_CUT = 'C/20'
DS_MODE = 'mean'
R_PTS = 32                      # PyBaMM radial resolution of the reference
N_DENSE = 51                    # the common scoring grid (--n-dense to change)
DST_CRATE, DST_SOC_START, DST_N_REP = 3.0, 0.95, 70

RUN_BASE = os.path.join(SM_ROOT, 'runs', 'v2_nondim')
OUT_DIR = os.path.join(SM_ROOT, 'runs', 'results')
REF_CACHE = os.path.join(OUT_DIR, 'ref_cache')
CSV = os.path.join(OUT_DIR, 'results.csv')
FIELD_NPZ = os.path.join(OUT_DIR, 'fields')      # saved concentration fields for the figure


def field_path(mname, cell, protocol, T_c, el, tag=DS_MODE):
    """One saved field.

    The tag is part of the NAME. Without it the unprojected pass silently overwrote the projected
    fields of the same model and cell - same prediction slot, completely different prediction - and
    any figure drawn afterwards showed the wrong one. A model can also be a sub-artifact path like
    <run>/best_pi, so the separator is replaced rather than stripped.
    """
    name = f'{mname}__{LEGACY.get(cell, cell)}__{protocol}__{T_c:+.0f}C__{el}__{tag}.npz'
    return os.path.join(FIELD_NPZ, name.replace('/', '-').replace(chr(92), '-'))

def _csv_lock(timeout=900):
    """Exclusive lock around the results CSV, so concurrent scorers cannot erase each other."""
    import contextlib
    import time as _t

    @contextlib.contextmanager
    def _cm():
        lock = CSV + '.lock'
        os.makedirs(os.path.dirname(CSV), exist_ok=True)
        t0 = _t.time()
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                if _t.time() - t0 > timeout:
                    log.warning('results.csv lock held for %ds - taking it over', timeout)
                    break
                _t.sleep(0.5)
        try:
            yield
        finally:
            try:
                os.remove(lock)
            except OSError:
                pass
    return _cm()


XI_D = np.linspace(0.0, 1.0, N_DENSE)
_E = np.concatenate([[0.0], (XI_D[1:] + XI_D[:-1]) / 2, [1.0]])
W_VOL = _E[1:] ** 3 - _E[:-1] ** 3
W_VOL = W_VOL / W_VOL.sum()                       # shell-volume weights summing to 1


FV_GRID = False                    # True: score at PyBaMM's own radial nodes (set_fv_grid)


def set_fv_grid(m):
    """Score at the reference's own points: the m finite-volume cell centres (k + 1/2)/m of the PyBaMM
    particle, plus the surface r = 1, where PyBaMM gives the surface concentration exactly. Nothing is
    interpolated on the reference side; with `direct` the network is evaluated at the same radii."""
    global N_DENSE, XI_D, W_VOL, R_PTS, FV_GRID
    R_PTS, FV_GRID = int(m), True
    XI_D = np.r_[(np.arange(R_PTS) + 0.5) / R_PTS, 1.0]
    N_DENSE = len(XI_D)
    e = np.arange(R_PTS + 1) / R_PTS
    W_VOL = np.r_[e[1:] ** 3 - e[:-1] ** 3, 0.0]     # each centre stands for its shell; the surface for none
    W_VOL = W_VOL / W_VOL.sum()


def set_grid(n):
    """Re-point the common scoring grid (and its volume weights) at n points."""
    global N_DENSE, XI_D, W_VOL
    N_DENSE = int(n)
    XI_D = np.linspace(0.0, 1.0, N_DENSE)
    e = np.concatenate([[0.0], (XI_D[1:] + XI_D[:-1]) / 2, [1.0]])
    W_VOL = e[1:] ** 3 - e[:-1] ** 3
    W_VOL = W_VOL / W_VOL.sum()


# ------------------------------------------------------------------ PyBaMM references
def _ref_key(cell, protocol, T_c, dst_period=None, dst_n_rep=None):
    """The overrides only enter the key when they are used, so every reference cached before they
    existed keeps its name and does not have to be re-solved."""
    parts = [LEGACY.get(cell, cell), protocol, T_c, PERIOD_S, CV_CUT, R_PTS,
             DST_CRATE, DST_SOC_START, DST_N_REP, pybamm.__version__]
    if dst_period is not None or dst_n_rep is not None:
        parts += ['dst', dst_period, dst_n_rep]
    if FV_GRID:                            # stored on PyBaMM's own nodes, no interpolation
        parts += ['fvgrid', R_PTS]
    elif N_DENSE != 101:                   # references are stored on the scoring grid; 101-point
        parts += ['dense', N_DENSE]        # references keep their original cache names
    return hashlib.md5(json.dumps(parts).encode()).hexdigest()[:10]


def _to_dense(prof_nodes, surf):
    """PyBaMM finite-volume rows -> the dense xi grid.

    The concentration lives at cell centres (k+0.5)/M, not at linspace(0,1,M). xi = 1 is anchored
    with PyBaMM's own particle surface concentration and xi = 0 with the first cell, which is what
    the centre symmetry condition dc/dxi = 0 implies."""
    if FV_GRID:                            # the scoring points ARE the cell centres + the surface
        return np.vstack([prof_nodes, surf[None, :]])
    M = prof_nodes.shape[0]
    nodes = (np.arange(M) + 0.5) / M
    xi_ref = np.concatenate([[0.0], nodes, [1.0]])
    full = np.vstack([prof_nodes[:1], prof_nodes, surf[None, :]])
    return np.stack([np.interp(XI_D, xi_ref, full[:, k]) for k in range(full.shape[1])], axis=1)


def DST_profile(n_rep):
    d = np.ones([361])
    d[:17] = 0; d[17:] = 0.125; d[45:] = 0.25; d[57:] = -0.125; d[65:] = 0
    d[81:] = 0.125; d[105:] = 0.25; d[117:] = -0.125; d[125:] = 0
    d[141:] = 0.125; d[165:] = 0.25; d[177:] = -0.125; d[185:] = 0
    d[201:] = 0.125; d[237:] = 1; d[245:] = 0.625; d[269:] = -0.25
    d[277:] = 0.25; d[309:] = -0.429; d[317:] = 0
    cur = d
    for _ in range(n_rep - 1):
        cur = np.concatenate((cur, d[1:]), axis=0)
    return np.arange(len(cur), dtype=float), cur


def build_reference(cell, protocol, T_c=T_C, dst_period=None, dst_n_rep=None):
    """(t, I_step, prof_dense{anode,cathode}, T_k) for one (cell, protocol, T). Cached on disk.

    dst_period: sample DST on a uniform grid of this many seconds instead of only at the current
    edges. The DST profile is defined at 1 Hz on integer seconds, so a period of 1 reproduces every
    edge exactly and gives a reference a BMS would actually see - long stretches at a constant
    current, which the edge-sampled grid collapses into one step each.
    dst_n_rep: repetitions of the 361 s DST cycle, overriding DST_N_REP (a 1 Hz grid at the full
    70 repetitions is 25 201 steps)."""
    os.makedirs(REF_CACHE, exist_ok=True)
    path = os.path.join(REF_CACHE, f'{_ref_key(cell, protocol, T_c, dst_period, dst_n_rep)}.npz')
    if os.path.exists(path):
        z = np.load(path)
        return z['t'], z['I'], {'anode': z['an'], 'cathode': z['ca']}, float(z['T_k'])

    pm = CELLS[cell][0]
    log.info('  PyBaMM: %s / %s / %+.0f C ...', cell, protocol, T_c)
    p = pybamm.ParameterValues(pm)
    T_k = 273.15 + T_c
    p['Initial temperature [K]'] = T_k
    p['Ambient temperature [K]'] = T_k
    cap = float(p['Nominal cell capacity [A.h]'])
    var_pts = {'x_n': 10, 'x_s': 10, 'x_p': 10, 'r_n': R_PTS, 'r_p': R_PTS}

    if protocol.startswith('CC-CV'):
        cr = float(protocol.split()[1].rstrip('C'))
        v_min, v_max = float(p['Lower voltage cut-off [V]']), float(p['Upper voltage cut-off [V]'])
        I_app = cap * cr
        p.set_initial_stoichiometries(1)
        exp = pybamm.Experiment([f'Discharge at {I_app} A until {v_min} V',
                                 f'Charge at {I_app} A until {v_max} V',
                                 f'Hold at {v_max} V until {CV_CUT}'], period=f'{PERIOD_S} seconds')
        sol = pybamm.Simulation(pybamm.lithium_ion.SPM(), parameter_values=p,
                                var_pts=var_pts, experiment=exp).solve()
        t = sol['Time [s]'].entries
        I = sol['Current [A]'].entries[:len(t) - 1]           # current held over each step
    else:                                                     # DST
        p.set_initial_stoichiometries(DST_SOC_START)
        p['Upper voltage cut-off [V]'] = float(p['Upper voltage cut-off [V]']) + 0.2
        t_f, frac = DST_profile(DST_N_REP if dst_n_rep is None else int(dst_n_rep))
        edges = np.flatnonzero(np.diff(frac) != 0) + 1
        t_grid = np.unique(np.concatenate([[t_f[0]], t_f[edges], [t_f[-1]]]))
        if dst_period is not None:          # uniform sampling; the edges are integer seconds, so
            t_grid = np.unique(np.concatenate([   # a period that divides them keeps them all
                np.arange(t_f[0], t_f[-1], float(dst_period)), t_grid]))
        seg = frac[np.searchsorted(t_f, t_grid[:-1], side='left')]
        p['Current function [A]'] = pybamm.Interpolant(t_f, frac * DST_CRATE * cap, pybamm.t,
                                                       interpolator='linear')
        sol = pybamm.Simulation(pybamm.lithium_ion.SPM(), parameter_values=p,
                                var_pts=var_pts).solve(t_eval=[t_f[0], t_f[-1]], t_interp=t_grid)
        t = sol['Time [s]'].entries
        I = seg[:len(t) - 1] * DST_CRATE * cap

    prof = {}
    for el, key in (('anode', 'negative'), ('cathode', 'positive')):
        nodes = np.mean(sol[f'{key.capitalize()} particle concentration'].entries, axis=1)
        surf = sol[f'X-averaged {key} particle surface concentration'].entries
        prof[el] = _to_dense(nodes[:, :len(t)], surf[:len(t)])

    np.savez_compressed(path, t=t, I=I, an=prof['anode'], ca=prof['cathode'], T_k=T_k)
    log.info('    %d steps, %.0f s -> cached', len(t) - 1, t[-1])
    return t, I, prof, T_k


# ------------------------------------------------------------------ roll-out + metrics
def roll(sur, chem, t, I, x0, T_k, timed=False, ds_mode=DS_MODE, direct=False):
    """Advance the surrogate over the protocol on ITS OWN sensor grid, then interpolate the whole
    history onto the dense grid. Returns (dense profile (N_DENSE, n_t), seconds per step)."""
    c = np.full(sur.n, float(x0))
    out = [np.full(len(XI_D), float(x0)) if direct else c]
    t0 = time.perf_counter() if timed else None
    for j in range(len(t) - 1):
        if direct:                        # the network evaluated at the scoring radii themselves
            c, at, _ = sur.step_series_at(c, float(I[j]), float(t[j + 1] - t[j]), chem, XI_D, T_k, ds_mode=ds_mode)
            out.append(at)
            continue
        c = sur.step(c, float(I[j]), float(t[j + 1] - t[j]), chem, T_k, ds_mode=ds_mode)
        out.append(c)
    dt_step = ((time.perf_counter() - t0) / max(len(t) - 1, 1)) if timed else np.nan
    hist = np.array(out).T                                    # (n_sensors, n_t)
    if direct:
        return hist, dt_step
    dense = np.stack([np.interp(XI_D, sur.sensors, hist[:, k]) for k in range(hist.shape[1])], axis=1)
    return dense, dt_step


def roll_adaptive(sur, chem, t, I, x0, T_k, tau_target, timed=False, ds_mode=DS_MODE, direct=False):
    """The deployment rule: one network call per WINDOW, not per protocol sample.

    `Surrogate.plan_window` grows the window until the accumulated tau would pass `tau_target` or
    the current jumps, whichever comes first; the window is fed the mean current over its samples
    (the exact equivalent constant current, since the reference holds I over each sample), and
    `Surrogate.step_series` reports the profile at every protocol sample inside it from the SAME
    window - one extra trunk evaluation each, never fed back into the state. So the output has the
    protocol's full time resolution while the state advances in long steps.

    Same return as `roll`, except the time is per NETWORK CALL; the third value is the mean number
    of protocol samples per call."""
    c = np.full(sur.n, float(x0))
    out = [np.full(len(XI_D), float(x0)) if direct else c]
    t0 = time.perf_counter() if timed else None
    j, calls = 0, 0
    while j < len(t) - 1:
        sto = sur._sto(c, ds_mode)
        m = sur.plan_window(chem, t[j:], I[j:], T_k, sto, tau_target=tau_target)
        span = float(t[j + m] - t[j])
        fracs = [(t[i] - t[j]) / span for i in range(j + 1, j + m)] if m > 1 else []
        if direct:                        # reports evaluated by the network at the scoring radii
            c, c_out, mid = sur.step_series_at(c, float(np.mean(I[j:j + m])), span, chem, XI_D, T_k,
                                               ds_mode=ds_mode, fracs=fracs)
            out.extend(mid)
            out.append(c_out)
        else:
            c, mid = sur.step_series(c, float(np.mean(I[j:j + m])), span, chem, T_k,
                                     ds_mode=ds_mode, fracs=fracs)
            out.extend(mid)
            out.append(c)
        j += m
        calls += 1
    dt_call = ((time.perf_counter() - t0) / max(calls, 1)) if timed else np.nan
    hist = np.array(out).T
    if direct:
        return hist, dt_call, (len(t) - 1) / max(calls, 1)
    dense = np.stack([np.interp(XI_D, sur.sensors, hist[:, k]) for k in range(hist.shape[1])], axis=1)
    return dense, dt_call, (len(t) - 1) / max(calls, 1)


def metrics(ref, pred):
    """All errors as a percentage of the reference stoichiometry range, on the dense grid."""
    d = pred - ref
    rng = float(ref.max() - ref.min())
    if rng <= 0:
        rng = 1.0
    soc_r = 3.0 * np.trapz(ref * (XI_D ** 2)[:, None], XI_D, axis=0)
    soc_p = 3.0 * np.trapz(pred * (XI_D ** 2)[:, None], XI_D, axis=0)
    return dict(
        pct_surf=100 * float(np.sqrt(np.mean(d[-1] ** 2))) / rng,
        pct_full=100 * float(np.sqrt(np.mean(d ** 2))) / rng,
        pct_full_vol=100 * float(np.sqrt(np.mean(np.sum(W_VOL[:, None] * d ** 2, axis=0)))) / rng,
        pct_soc=100 * float(np.sqrt(np.mean((soc_p - soc_r) ** 2))) / rng,
        max_pct_surf=100 * float(np.abs(d[-1]).max()) / rng,
        sto_range=rng,
    )


# ------------------------------------------------------------------ driver
def _is_run(name):
    d = os.path.join(RUN_BASE, name)
    return (os.path.exists(os.path.join(d, 'surrogate_manifest.json'))
            and os.path.exists(os.path.join(d, 'model.pt')))


def discover_models(only=None):
    """Scored models. Without `only`, the top-level run folders.

    With `only`, a name may also be a SUB-ARTIFACT path such as "<run>/best_pi" - the checkpoint
    selected on the held-out physics loss, which train_nd writes as a nested folder. The listing
    below only ever sees top-level entries, so an explicitly named path is checked directly;
    without this, asking for the pi-selected model failed with "no finished run named".
    """
    if only:
        runs = [m for m in only if _is_run(m)]
        missing = [m for m in only if m not in runs]
        if missing:
            raise SystemExit(f'no finished run named {missing} in {RUN_BASE}')
        return runs
    return [n for n in sorted(os.listdir(RUN_BASE)) if _is_run(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--cells', nargs='*', default=None)
    ap.add_argument('--protocols', nargs='*', default=None)
    ap.add_argument('--temperatures', nargs='*', type=float, default=None,
                    help='isothermal run temperatures in C (default %s)' % T_LIST)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--n-dense', type=int, default=N_DENSE, help='points of the common scoring grid')
    ap.add_argument('--csv', default=None, help='results file (default runs/results/results.csv)')
    ap.add_argument('--no-timing', action='store_true')
    ap.add_argument('--ds-mode', default=DS_MODE, choices=['ref', 'mean', 'surface'],
                    help="which stoichiometry D_s is evaluated at each step; only matters for a "
                         "parameter set whose D_s depends on stoichiometry")
    ap.add_argument('--save-fields', nargs='*', default=['Mohtat2020'],
                    help='cells whose full concentration field is saved for the figures '
                         '(no argument = none). A row already in the CSV is still recomputed when '
                         'its field file is missing, so asking for a new cell does not need --force '
                         'and does not recompute anything else.')
    ap.add_argument('--no-projection', action='store_true',
                    help='score WITHOUT the lithium-balance projection, which is a uniform '
                         'shift: it pins the volume mean after every step, so it masks any '
                         'error in the MEAN while leaving shape error untouched. Rows are '
                         'tagged ds_mode="<mode>-noproj" so both versions coexist.')
    ap.add_argument('--fv-grid', type=int, default=None, metavar='M',
                    help='score at the PyBaMM reference own radial nodes: M finite volumes, i.e. M cell centres plus '
                         'the surface (the reference is solved with M volumes). Rows tagged "<tag>-fv<M>".')
    ap.add_argument('--direct', action='store_true',
                    help='evaluate the network directly at the scoring radii instead of interpolating its '
                         'sensor values onto them (the state still advances on the sensors). Rows are '
                         'tagged "<tag>-direct".')
    ap.add_argument('--tau-target', type=float, default=None,
                    help='deploy with the adaptive window rule (Surrogate.plan_window): one '
                         'network call per window, grown until tau would pass this value or the '
                         'current jumps, with every protocol sample inside it reported from the '
                         'same window. 0.03 is the value that was measured best. Rows '
                         'are tagged ds_mode="<mode>-adapt<tau>" so they coexist with the '
                         'one-call-per-sample scores.')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    pybamm.set_logging_level('ERROR')
    global CSV
    if a.fv_grid:
        set_fv_grid(a.fv_grid)
    elif a.n_dense != N_DENSE:
        set_grid(a.n_dense)
    if a.csv:
        CSV = os.path.abspath(a.csv)

    # single CPU core, so the per-step cost is a number the paper can quote
    torch.set_num_threads(1)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    models = discover_models(a.models)
    ds_mode = a.ds_mode                    # FUNCTIONAL: which stoichiometry D_s is read at.
    # The CSV tag is a SEPARATE string. ds_mode is passed straight into Surrogate.step, which only
    # accepts 'ref' | 'mean' | 'surface', so the projection flag must not be folded into it - it is
    # recorded as a suffix on the stored tag instead, which lets projected and unprojected scores
    # of the same model coexist as separate rows.
    ds_tag = (ds_mode + ('-noproj' if a.no_projection else '')
              + ('-adapt%g' % a.tau_target if a.tau_target else '') + ('-direct' if a.direct else '') + ('-fv%d' % a.fv_grid if a.fv_grid else ''))
    cells = ([c for c in CELLS if any(k.lower() in c.lower() for k in a.cells)] if a.cells
             else HEADLINE_CELLS)
    protocols = a.protocols or PROTOCOLS
    temps = a.temperatures if a.temperatures is not None else T_LIST
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIELD_NPZ, exist_ok=True)
    save_cells = set(a.save_fields or [])

    done = pd.read_csv(CSV) if (os.path.exists(CSV) and not a.force) else pd.DataFrame()
    if not done.empty and 'ds_mode' not in done.columns:
        done['ds_mode'] = DS_MODE                       # rows written before the flag existed
    if not done.empty:
        done['cell'] = done['cell'].replace(FROM_LEGACY)  # results written with the old cell names
    key_cols = ['model', 'cell', 'protocol', 'electrode', 'ds_mode', 'T_C']
    have = set() if done.empty else set(map(tuple, done[key_cols].values))

    log.info('models   : %s', ', '.join(models))
    log.info('cells    : %s', ', '.join(cells))
    log.info('protocols: %s', ', '.join(protocols))
    log.info('temps    : %s C', ', '.join('%+.0f' % t for t in temps))
    log.info('scoring  : dense grid of %d points, PyBaMM r_pts=%d, ds_mode=%s\n',
             N_DENSE, R_PTS, ds_tag)

    rows = []
    for cell in cells:
        pm = CELLS[cell][0]
        chems = {'anode': Chemistry.from_pybamm(pm, 'Negative'),
                 'cathode': Chemistry.from_pybamm(pm, 'Positive')}
        for protocol in protocols:
          for T_c in temps:
            t, I, ref, T_k = build_reference(cell, protocol, T_c)
            for mname in models:
                sur = Surrogate.from_run(os.path.join(RUN_BASE, mname),
                                         mass_correction=not a.no_projection)
                for el in ('anode', 'cathode'):
                    want_field = cell in save_cells and not os.path.exists(
                        field_path(mname, cell, protocol, T_c, el, ds_tag))
                    if (mname, cell, protocol, el, ds_tag, T_c) in have and not want_field:
                        continue
                    scored = (mname, cell, protocol, el, ds_tag, T_c) in have
                    if a.tau_target:
                        pred, dt_step, per_call = roll_adaptive(
                            sur, chems[el], t, I, ref[el][0, 0], T_k, a.tau_target,
                            timed=not a.no_timing, ds_mode=ds_mode, direct=a.direct)
                    else:
                        pred, dt_step = roll(sur, chems[el], t, I, ref[el][0, 0], T_k,
                                             timed=not a.no_timing, ds_mode=ds_mode, direct=a.direct)
                        per_call = 1.0
                    m = metrics(ref[el], pred)
                    if scored:                      # recomputed only to write the field file
                        if cell in save_cells:
                            np.savez_compressed(field_path(mname, cell, protocol, T_c, el, ds_tag),
                                                t=t, I=I, xi=XI_D, ref=ref[el], pred=pred)
                        continue
                    phi_max = max(abs(chems[el].flux(float(i), T_k)) for i in I)
                    rows.append(dict(model=mname, cell=cell, param_set=pm, format=CELLS[cell][1],
                                     protocol=protocol, electrode=el, T_C=T_c,
                                     n_steps=len(t) - 1, duration_s=float(t[-1]),
                                     phi_max=phi_max, ms_per_step=1e3 * dt_step,
                                     n_sensors=sur.n, ds_mode=ds_tag, samples_per_call=per_call,
                                     ds_x_ratio=chems[el].ds_x_ratio, **m))
                    log.info('  %-22s %-18s %-11s %+5.0fC %-7s  surf %.3f %%  vol %.3f %%  soc %.3f %%  %.2f ms/step',
                             mname, cell, protocol, T_c, el, m['pct_surf'], m['pct_full_vol'],
                             m['pct_soc'], 1e3 * dt_step)
                    if cell in save_cells:
                        np.savez_compressed(field_path(mname, cell, protocol, T_c, el, ds_tag),
                                            t=t, I=I, xi=XI_D, ref=ref[el], pred=pred)

    if not rows:
        log.info('nothing new to compute (use --force to recompute)')
        return
    new = pd.DataFrame(rows)
    # RE-READ under the lock and merge, rather than writing the snapshot taken at start-up.
    # Two scorers running at once - the pipeline's own projected pass and a manual --no-projection
    # pass, say - each read the CSV before the other wrote, and whichever finished last erased the
    # other's rows. That silently deleted a full 150-row projected scoring on 2026-09-24.
    with _csv_lock():
        cur = pd.read_csv(CSV) if os.path.exists(CSV) else pd.DataFrame()
        if not cur.empty:
            cur['cell'] = cur['cell'].replace(FROM_LEGACY)
        out = new if cur.empty else pd.concat([cur, new], ignore_index=True)
        kc = [c for c in ('model', 'cell', 'protocol', 'electrode', 'ds_mode', 'T_C')
              if c in out.columns]
        before = len(out)
        out = out.drop_duplicates(subset=kc, keep='last')
        if before != len(out):
            log.info('merged: dropped %d duplicate rows', before - len(out))
        out.to_csv(CSV, index=False)
    log.info('\n%d new rows -> %s (%d total)', len(new), CSV, len(out))

    piv = new.pivot_table(index=['cell', 'protocol'], columns=['electrode', 'T_C'],
                          values='pct_surf', aggfunc='mean')
    log.info('\nmean surface error [%% of range], averaged over models:\n%s', piv.round(3).to_string())


if __name__ == '__main__':
    main()
