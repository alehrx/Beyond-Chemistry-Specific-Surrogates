"""
cycles_vs_dfn.py
================
The same MIONet as the particle solver of three cell models, against one plant: PyBaMM's DFN for the
cell of Ai2020 at 25 C, over complete cycles (constant-current discharge, constant-current charge,
constant-voltage hold to C/20) at 1C, 2C and 3C.

    SPM          srcs_EM.Cell(electrolyte=False): two particles + lumped electrolyte resistance
    SPMe         srcs_EM.Cell(electrolyte=True):  two particles + 4-mode electrolyte
    multi-slice  srcs_EM.MultiSlice, 10 particles per electrode + 8-mode electrolyte

Every model is driven by the plant's current, averaged over each 5-s interval from the plant's own
charge counter (exact for the constant-voltage phase as well), and steps every 5 s.

Figure 1  fig_cycles_dfn          left: terminal voltage (top) and current in C-rate (bottom) against
                                  time normalised by the cycle duration; right: voltage error
                                  model - plant
Figure 2  fig_surface_dfn         surface stoichiometry across the electrode thickness over the 3C
                                  cycle: DFN plant, multi-slice model, relative difference (% of the
                                  plant's surface-stoichiometry range); anode and cathode; applied
                                  current below

Ai2020 ships its electrolyte diffusivity in cm2/s under an m2/s label; plant and models both use the
corrected value. Writes runs/cell_models/ai2020_NNN/ (metrics.csv, config.json, metadata.json) and
the figures to figures/ (or MIONET_FIG_DIR).

    python scripts/cell_models/cycles_vs_dfn.py
"""
import datetime
import json
import logging
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

mpl.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'axes.prop_cycle': mpl.cycler('color', ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7', '#56B4E9']),
    'lines.linewidth': 1.5, 'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'xtick.major.size': 4, 'ytick.major.size': 4, 'xtick.minor.size': 2, 'ytick.minor.size': 2,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
    'lines.markersize': 4, 'errorbar.capsize': 3, 'axes.xmargin': 0.02, 'axes.ymargin': 0.02,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.dpi': 300,
    **({'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
        'pgf.texsystem': 'pdflatex', 'pgf.rcfonts': False}
       if __import__('shutil').which('latex') else {'text.usetex': False, 'mathtext.fontset': 'cm'}),
})
mm = 1 / 25.4

HERE = os.path.dirname(os.path.abspath(__file__))
SM = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, SM)
import pybamm                                                     # noqa: E402
from srcs_EM import Cell, MultiSlice                            # noqa: E402
from srcs_EM.cell import CellState                              # noqa: E402

MODEL = os.path.join(SM, 'runs', 'v2_nondim', 'trained_model_s42_lbfgs')
PARAM_SET, T_C, PERIOD_S, CV_CUT = 'Ai2020', 25.0, 5.0, 'C/20'   # 5 s: with 10 thin slices the
# currents frozen over a 20-s step make the coupling unstable at the 3C reversal
RATES, FIELD_RATE = (1.0, 2.0, 3.0), 3.0
N_SLICES, SPME_MODES, SLICE_MODES, N_FV = 10, 4, 8, 150
DIFF_WINDOW_S = 60.0   # fig_surface_dfn: the difference panel is a moving average over this window, so a
                       # single-sample spike at the current reversal does not set the colour scale
OUT = os.path.join(SM, 'runs', 'cell_models')
FIG = os.environ.get('MIONET_FIG_DIR') or os.path.join(SM, 'figures')        # override: paper folder
TEX = mpl.rcParams['text.usetex']
PLANT_C = '0.1'
COL = {'SPM': '#0072B2', 'SPMe': '#009E73', 'multi-slice': '#D55E00'}
RATE_LS = {1.0: (0, (1.2, 1.2)), 2.0: (0, (4, 1.5)), 3.0: '-'}
log = logging.getLogger(__name__)


def next_dir():
    os.makedirs(OUT, exist_ok=True)
    i = 1
    while os.path.exists(os.path.join(OUT, 'ai2020_%03d' % i)):
        i += 1
    d = os.path.join(OUT, 'ai2020_%03d' % i)
    os.makedirs(d)
    return d


def plant(c_rate, de_scale):
    """PyBaMM DFN cycle; the current over each interval is the mean from the charge counter."""
    p = pybamm.ParameterValues(PARAM_SET)
    base = p['Electrolyte diffusivity [m2.s-1]']
    if de_scale != 1.0:
        p.update({'Electrolyte diffusivity [m2.s-1]': lambda c_e, T: de_scale * base(c_e, T)})
    T_K = 273.15 + T_C
    p.update({'Ambient temperature [K]': T_K, 'Initial temperature [K]': T_K})
    q = float(p['Nominal cell capacity [A.h]'])
    v_min, v_max = float(p['Lower voltage cut-off [V]']), float(p['Upper voltage cut-off [V]'])
    p.set_initial_stoichiometries(1.0)
    exp = pybamm.Experiment([f'Discharge at {c_rate * q} A until {v_min} V',
                             f'Charge at {c_rate * q} A until {v_max} V',
                             f'Hold at {v_max} V until {CV_CUT}'], period=f'{PERIOD_S} seconds')
    sol = pybamm.Simulation(pybamm.lithium_ion.DFN(), parameter_values=p, experiment=exp).solve()
    t = sol['Time [s]'].entries
    # an experiment repeats the time point at every step boundary (end of one step, start of the
    # next); those zero-length intervals would turn the charge counter into spurious currents
    keep = np.r_[True, np.diff(t) > 1e-6]
    t = t[keep]
    Q = sol['Discharge capacity [A.h]'].entries[keep] * 3600.0
    I_mean = np.diff(Q) / np.diff(t)                     # exact mean current of every interval
    c_n = float(p['Maximum concentration in negative electrode [mol.m-3]'])
    c_p = float(p['Maximum concentration in positive electrode [mol.m-3]'])
    return dict(t=t, V=sol['Voltage [V]'].entries[keep], I=sol['Current [A]'].entries[keep], I_mean=I_mean, q=q,
                T_K=T_K, x_n=sol['x_n [m]'].entries[:, 0], x_p=sol['x_p [m]'].entries[:, 0],
                s_n=sol['Negative particle surface stoichiometry'].entries[:, keep],
                s_p=sol['Positive particle surface stoichiometry'].entries[:, keep],
                x_n0=float(p['Initial concentration in negative electrode [mol.m-3]']) / c_n,
                x_p0=float(p['Initial concentration in positive electrode [mol.m-3]']) / c_p)


def phases(r):
    """Masks of the discharge, constant-current charge and constant-voltage samples of the plant."""
    I = r['I']
    q = r['q']
    dis = I > 0
    cv = (I < 0) & (np.abs(I) < 0.98 * np.abs(I[I < 0]).max())
    return dict(discharge=dis, charge=(I < 0) & ~cv, cv=cv)


def rollout(model, r):
    """Step with the MEAN current of every interval (exact charge); evaluate the voltage at every
    sample with the plant's INSTANTANEOUS current there, as the plant reports it."""
    t, I_inst, I_mean, T_K = r['t'], r['I'], r['I_mean'], r['T_K']
    V = np.zeros(len(t))
    if isinstance(model, MultiSlice):
        st = model.initial_state(r['x_n0'], r['x_p0'])
        sn, sp = np.zeros((len(t), model.N['n'])), np.zeros((len(t), model.N['p']))
        for k in range(len(t)):
            out = model.split(st, float(I_inst[k]), T_K)
            V[k], sn[k], sp[k] = out['V'], out['sto_n'], out['sto_p']
            if k + 1 < len(t):
                st = model.step(st, float(I_mean[k]), float(t[k + 1] - t[k]), T_K)
        return dict(t=t, V=V, sto_n_surf=sn, sto_p_surf=sp)
    n = model.sur.n
    q0 = np.zeros(model.elyte.n_modes) if model.elyte is not None else np.zeros(0)
    st = CellState(np.full(n, r['x_n0']), np.full(n, r['x_p0']), q0)
    for k in range(len(t)):
        V[k] = model.terminal(st, float(I_inst[k]), T_K)['V']
        if k + 1 < len(t):
            st = model.step(st, float(I_mean[k]), float(t[k + 1] - t[k]), T_K)
    return dict(t=t, V=V)


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    pybamm.set_logging_level('ERROR')
    run_dir = next_dir()
    spm = Cell.from_run(MODEL, PARAM_SET, electrolyte=False)
    spme = Cell.from_run(MODEL, PARAM_SET, electrolyte=True, n_modes=SPME_MODES, tdf_mode='quad')
    ms = MultiSlice.from_run(MODEL, PARAM_SET, n_slices=N_SLICES, n_modes=SLICE_MODES, n_fv=N_FV)
    de_scale = ms.ely.de_scale
    ref, res, rows = {}, {}, []
    for cr in RATES:
        r = ref[cr] = plant(cr, de_scale)
        for name, model in (('SPM', spm), ('SPMe', spme), ('multi-slice', ms)):
            res[(cr, name)] = rollout(model, r)
        ph = phases(r)
        for name in COL:
            e = 1e3 * (res[(cr, name)]['V'] - r['V'])
            row = dict(c_rate=cr, model=name, rmse_mV=float(np.sqrt(np.mean(e ** 2))),
                       max_mV=float(np.abs(e).max()), n_steps=len(r['t']) - 1, duration_min=r['t'][-1] / 60)
            for p_name, m in ph.items():
                row['rmse_%s_mV' % p_name] = float(np.sqrt(np.mean(e[m] ** 2)))
            rows.append(row)
        log.info('%gC cycle, %d steps over %.0f min: %s', cr, len(r['t']) - 1, r['t'][-1] / 60,
                 ' | '.join('%s %.1f mV (dis %.1f, chg %.1f, CV %.1f)' % (x['model'], x['rmse_mV'],
                            x['rmse_discharge_mV'], x['rmse_charge_mV'], x['rmse_cv_mV'])
                            for x in rows if x['c_rate'] == cr))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run_dir, 'metrics.csv'), index=False)
    os.makedirs(FIG, exist_ok=True)
    fig_cycles(ref, res).savefig(os.path.join(FIG, 'fig_cycles_dfn.pdf'))
    surf = fig_surface(ref[FIELD_RATE], res[(FIELD_RATE, 'multi-slice')], ms)
    surf.savefig(os.path.join(FIG, 'fig_surface_dfn.pdf'))
    json.dump(dict(param_set=PARAM_SET, T_C=T_C, period_s=PERIOD_S, cv_cut=CV_CUT, rates=RATES,
                   field_rate=FIELD_RATE, n_slices=N_SLICES, spme_modes=SPME_MODES, slice_modes=SLICE_MODES,
                   n_fv=N_FV, de_scale=de_scale, model=os.path.relpath(MODEL, SM),
                   pybamm_version=pybamm.__version__),
              open(os.path.join(run_dir, 'config.json'), 'w'), indent=2)
    json.dump(dict(date=datetime.datetime.now().isoformat(timespec='seconds'),
                   description='MIONet in SPM / SPMe / multi-slice against a PyBaMM DFN plant, full cycles',
                   dataset='PyBaMM DFN, Ai2020 25 C, 1/2/3C CC-CV cycles (D_e unit-corrected)'),
              open(os.path.join(run_dir, 'metadata.json'), 'w'), indent=2)
    print(df.pivot_table(index='c_rate', columns='model', values='rmse_mV').round(2).to_string())
    log.info('written %s and figures fig_cycles_dfn, fig_surface_dfn', run_dir)


def fig_cycles(ref, res):
    fig = plt.figure(figsize=(170 * mm, 95 * mm))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2.3, 1], width_ratios=[1.15, 1], hspace=0.08, wspace=0.28,
                  left=0.08, right=0.98, top=0.86, bottom=0.12)
    ax_v = fig.add_subplot(gs[0, 0])
    ax_i = fig.add_subplot(gs[1, 0], sharex=ax_v)
    ax_e = fig.add_subplot(gs[:, 1])
    for cr in RATES:
        r = ref[cr]
        tn = r['t'] / r['t'][-1]
        ax_v.plot(tn, r['V'], color=PLANT_C, lw=1.3, ls=RATE_LS[cr])
        ax_i.plot(tn[:-1], r['I_mean'] / r['q'], color=PLANT_C, lw=1.0, ls=RATE_LS[cr], drawstyle='steps-post')
        for name, c in COL.items():
            ax_v.plot(tn, res[(cr, name)]['V'], color=c, lw=0.9, ls=RATE_LS[cr], alpha=0.85)
            ax_e.plot(tn, 1e3 * (res[(cr, name)]['V'] - r['V']), color=c, lw=0.9, ls=RATE_LS[cr])
    ax_v.set_ylabel('terminal voltage [V]')
    plt.setp(ax_v.get_xticklabels(), visible=False)
    ax_i.axhline(0, color='0.8', lw=0.6, zorder=0)
    ax_i.set_ylabel(r'$I$ [C-rate]' if TEX else 'I [C-rate]')
    ax_i.set_xlabel('time / cycle duration')
    ax_e.axhline(0, color='0.8', lw=0.6, zorder=0)
    ax_e.set_xlabel('time / cycle duration')
    ax_e.set_ylabel('voltage error [mV]')
    for ax in (ax_v, ax_i, ax_e):
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(0, 1)
    for ax, l, t in ((ax_v, '(a)', 'Terminal voltage and applied current'), (ax_e, '(b)', 'Voltage error')):
        ax.set_title((r'\textbf{%s} %s' if TEX else '%s %s') % (l, t), loc='left', fontsize=9)
    h = [Line2D([], [], color=PLANT_C, lw=1.3, label='PyBaMM DFN (plant)')]
    h += [Line2D([], [], color=c, lw=1.2, label='MIONet in %s' % n) for n, c in COL.items()]
    h += [Line2D([], [], color='0.4', lw=1.0, ls=RATE_LS[cr], label='%gC' % cr) for cr in RATES]
    fig.legend(handles=h, loc='upper center', ncol=7, fontsize=7.5, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.0, columnspacing=1.0)
    return fig


def fig_surface(r, o, ms):
    """Surface stoichiometry over (t, x) in each electrode: DFN, multi-slice, |difference|."""
    tm = r['t'] / 60
    fig = plt.figure(figsize=(170 * mm, 112 * mm))
    wr = [1.0, 0.13, 1.0, 0.035, 0.42, 1.0, 0.035]
    gs = GridSpec(3, 7, figure=fig, width_ratios=wr, height_ratios=[1.0, 1.0, 0.42], hspace=0.18,
                  wspace=0.05, left=0.07, right=0.93, top=0.93, bottom=0.10)
    L_n, L_p = ms.chem['n'].L, ms.chem['p'].L
    for row, (key, name, xr, s_ref, s_mod, L) in enumerate(
            (('n', 'anode', r['x_n'], r['s_n'], o['sto_n_surf'], L_n),
             ('p', 'cathode', r['x_p'] - r['x_p'][0] + 0.5 * (r['x_p'][1] - r['x_p'][0]), r['s_p'],
              o['sto_p_surf'], L_p))):
        N = s_mod.shape[1]
        edges = np.linspace(0.0, L, N + 1)
        k = np.clip(np.searchsorted(edges, xr, side='right') - 1, 0, N - 1)
        s_mod_x = s_mod[:, k].T                                  # slice value at every plant point
        # the slice model carries one value per slice: compare it with the DFN averaged over that slice
        s_ref_slice = np.stack([s_ref[k == j].mean(axis=0) for j in range(N)], axis=1)   # (n_t, N)
        span = float(s_ref.max() - s_ref.min())                 # the plant's range in this electrode
        diff = 100.0 * np.abs(s_mod - s_ref_slice) / span                      # (n_t, N) per sample
        n_w = max(1, int(round(DIFF_WINDOW_S / float(np.median(np.diff(r['t']))))))
        diff = uniform_filter1d(diff, size=n_w, axis=0, mode='nearest')        # windowed in time
        diff_x = diff[:, k].T
        lo, hi = float(min(s_ref.min(), s_mod.min())), float(max(s_ref.max(), s_mod.max()))
        y_um = xr * 1e6
        panels = ((0, s_ref, 'PyBaMM DFN', 'viridis', lo, hi),
                  (2, s_mod_x, 'MIONet, %d slices' % N, 'viridis', lo, hi),
                  (5, diff_x, 'Relative difference, %.0f-s average' % DIFF_WINDOW_S, 'magma', 0, None))
        for col, Z, title, cmap, vmin, vmax in panels:
            ax = fig.add_subplot(gs[row, col])
            im = ax.pcolormesh(tm, y_um, Z, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
            if row == 0:
                ax.set_title(title, fontsize=9, loc='left')
            if col != 0:
                ax.tick_params(labelleft=False)
            if col == 0:
                ax.set_ylabel('%s, $x$ [$\\mu$m]' % name if TEX else '%s, x [um]' % name)
            if row == 0:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('time [min]')
            ax.minorticks_off()
            if col in (2, 5):
                cb = fig.colorbar(im, cax=fig.add_subplot(gs[row, col + 1]))
                cb.ax.minorticks_off()
                cb.ax.tick_params(labelsize=7)
                cb.set_label(('surface stoich.' if col == 2 else
                              (r'rel. difference [\%]' if TEX else 'rel. difference [%]')), fontsize=8)
        log.info('%s surface stoichiometry, %gC, slice vs DFN slice average: RMSE %.2e (%.2f %% of range), '
                 'max %.2e (%.2f %% of range)', name, FIELD_RATE,
                 float(np.sqrt(np.mean((s_mod - s_ref_slice) ** 2))),
                 100 * float(np.sqrt(np.mean((s_mod - s_ref_slice) ** 2))) / span,
                 float(np.abs(s_mod - s_ref_slice).max()), 100 * float(np.abs(s_mod - s_ref_slice).max()) / span)
    # the applied current under every column, on the same time axis as the fields above
    cr = r['I_mean'] / r['q']
    pad = 0.1 * float(np.max(np.abs(cr)))
    for col in (slice(None),):
        ax = fig.add_subplot(gs[2, col])
        ax.plot(tm[:-1], cr, color='k', lw=1.0, drawstyle='steps-post')
        ax.axhline(0, color='0.75', lw=0.6, zorder=0)
        ax.set_xlim(tm[0], tm[-1])
        ax.set_ylim(float(cr.min()) - pad, float(cr.max()) + pad)
        ax.set_xlabel('time [min]')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylabel(r'$I$ [C-rate]' if TEX else 'I [C-rate]')
    return fig


if __name__ == '__main__':
    main()
