"""
fig_fields.py
=============
Concentration fields of one complete cycle (the paper's case: Mohtat2020, 0 C, CC-CV 1C), reported
model trained_model_s42_lbfgs (seed 42), rolled out with the adaptive window rule exactly as in the scoring.

    (a) anode | cathode, three rows each: PyBaMM, MIONet, relative error (% of the electrode's
        stoichiometry range, the normalisation of every error in the paper), over (t, r)
    (b) applied current, with the four instants of (c) marked
    (c) radial profiles at those instants: PyBaMM lines, MIONet markers

Draws the paper's case (CASES); set CELL, PROTOCOL, T_C and STEM in the environment for another one. Writes figures/<STEM>.pdf (or MIONET_FIG_DIR).
"""
import os
import sys
import logging

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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
sys.path.insert(0, os.path.join(SM, 'srcs'))
import evaluation as rr                          # noqa: E402
from srcs_EM import Surrogate, Chemistry       # noqa: E402

RUN = os.path.join(SM, 'runs', 'v2_nondim', 'trained_model_s42_lbfgs')
# the paper's three field figures: (cell, protocol, T [C], output name)
CASES = [('Mohtat2020', 'CC-CV 1C', 0.0, 'fig_fields_mohtat2020')]
if os.environ.get('CELL'):                       # one case from the environment instead
    CASES = [(os.environ['CELL'], os.environ.get('PROTOCOL', 'CC-CV 1C'), float(os.environ.get('T_C', '25')),
              os.environ.get('STEM', 'fig_fields_custom'))]
TAU_TARGET = 0.05                                   # window rule: tau <= 0.05 or a 5 % current change
N_GRID = 51                                         # common scoring grid, as in srcs/evaluation.py
rr.set_grid(N_GRID)
FIG = os.environ.get('MIONET_FIG_DIR') or os.path.join(SM, 'figures')        # override: paper folder
SNAP_FRAC = (0.15, 0.42, 0.75, 0.95)
MARKERS = ('o', 's', '^', 'D')
SNAP_COL = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')
TEX = mpl.rcParams['text.usetex']
log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    sur = Surrogate.from_run(RUN)
    for case in CASES:
        draw(sur, *case)


def draw(sur, CELL, PROTOCOL, T_C, STEM):
    import pybamm
    t, I, ref, T_k = rr.build_reference(CELL, PROTOCOL, T_C)
    pm = rr.CELLS[CELL][0]
    cap = float(pybamm.ParameterValues(pm)['Nominal cell capacity [A.h]'])
    log.info('--- %s %s %+.0f C', CELL, PROTOCOL, T_C)
    pred = {}
    for el, side in (('anode', 'Negative'), ('cathode', 'Positive')):
        pred[el], _, spc = rr.roll_adaptive(sur, Chemistry.from_pybamm(pm, side), t, I, ref[el][0, 0], T_k,
                                            TAU_TARGET)
        m = rr.metrics(ref[el], pred[el])
        log.info('%s: surface %.3f %%, full (vol) %.3f %%, %.0f network calls for %d samples', el,
                 m['pct_surf'], m['pct_full_vol'], (len(t) - 1) / spc, len(t) - 1)

    tm, r = t / 60.0, rr.XI_D
    snaps = [int(f * (len(t) - 1)) for f in SNAP_FRAC]
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(170 * mm, 205 * mm))
    # columns: anode field | its colour bar | gap | cathode field | its colour bar
    wr = [1.0, 0.035, 0.44, 1.0, 0.035]
    L, R = 0.085, 0.915
    g_top = GridSpec(3, 5, figure=fig, left=L, right=R, top=0.965, bottom=0.555, width_ratios=wr,
                     wspace=0.06, hspace=0.16)
    g_mid = GridSpec(1, 5, figure=fig, left=L, right=R, top=0.475, bottom=0.391, width_ratios=wr, wspace=0.06)
    g_bot = GridSpec(1, 5, figure=fig, left=L, right=R, top=0.315, bottom=0.075, width_ratios=wr, wspace=0.06)
    ax = {}
    for c0, key in ((0, 'a'), (3, 'c')):
        for row, rk in enumerate('PME'):
            ax[key + rk] = fig.add_subplot(g_top[row, c0])
        ax[key + 'cb1'] = fig.add_subplot(g_top[0:2, c0 + 1])
        ax[key + 'cb2'] = fig.add_subplot(g_top[2, c0 + 1])
        ax['p' + key] = fig.add_subplot(g_bot[0, c0])
    ax['cur'] = fig.add_subplot(g_mid[0, 0:4])
    lab = {'anode': r'$c_{s,\mathrm{n}}$' if TEX else 'c_s,n', 'cathode': r'$c_{s,\mathrm{p}}$' if TEX else 'c_s,p'}
    for el, key in (('anode', 'a'), ('cathode', 'c')):
        lo, hi = float(ref[el].min()), float(ref[el].max())
        err = 100.0 * np.abs(pred[el] - ref[el]) / (hi - lo)       # % of the stoichiometry range
        for row, Z, name, cmap, vmin, vmax in (('P', ref[el], 'PyBaMM', 'viridis', lo, hi),
                                              ('M', pred[el], 'MIONet', 'viridis', lo, hi),
                                              ('E', err, 'rel. error', 'magma', 0.0, None)):
            a = ax[key + row]
            im = a.pcolormesh(tm, r, Z, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
            for k, c in zip(snaps, SNAP_COL):
                a.axvline(tm[k], color=c, lw=0.8, ls=':')
            a.text(0.015, 0.93, name, transform=a.transAxes, va='top', fontsize=8, color='white',
                   bbox=dict(facecolor='black', alpha=0.35, edgecolor='none', pad=1.5))
            a.set_ylabel('$r$')
            a.set_yticks([0, 0.5, 1])
            a.minorticks_off()
            if row == 'E':
                a.set_xlabel('time [min]')
            if row == 'P':
                a.set_title(el.capitalize(), fontsize=9, loc='left')
            if row in ('M', 'E'):
                cb = fig.colorbar(im, cax=ax[key + ('cb1' if row == 'M' else 'cb2')])
                cb.set_label(lab[el] if row == 'M' else (r'rel. error [\%]' if TEX else 'rel. error [%]'))
                cb.ax.minorticks_off()
    a = ax['cur']
    a.plot(tm[:-1], I / cap, color='k', lw=0.8 if PROTOCOL.startswith('DST') else 1.0, drawstyle='steps-post')
    a.axhline(0, color='0.75', lw=0.6, zorder=0)
    pad = 0.1 * float(np.max(np.abs(I / cap)))            # gap between the curve and the axes
    y0, y1 = float(np.min(I / cap)) - pad, float(np.max(I / cap)) + pad
    a.set_ylim(y0, y1)
    for k, c, mk in zip(snaps, SNAP_COL, MARKERS):
        a.axvline(tm[k], color=c, lw=0.8, ls=':')
        a.plot(tm[k], y0, marker=mk, color=c, ms=5, clip_on=False, zorder=5)   # on the time axis
    a.set_xlabel('time [min]')
    a.set_ylabel(r'$I$ [C-rate]' if TEX else 'I [C-rate]')
    a.set_xlim(tm[0], tm[-1])
    for el, key in (('anode', 'pa'), ('cathode', 'pc')):
        a = ax[key]
        for k, c, mk in zip(snaps, SNAP_COL, MARKERS):
            a.plot(r, ref[el][:, k], color='0.35', lw=0.9)
            a.plot(r[::5], pred[el][::5, k], ls='none', marker=mk, mfc='none', mec=c, mew=1.0, ms=5,
                   label='$t = %.0f$ min' % tm[k] if TEX else 't = %.0f min' % tm[k])
        a.set_xlabel('$r$')
        a.set_ylabel('%s (%s)' % (lab[el], el))
        a.set_xlim(0, 1)
    h, l = ax['pa'].get_legend_handles_labels()
    h = [mpl.lines.Line2D([], [], color='0.35', lw=0.9)] + h
    l = ['PyBaMM'] + l
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    for key in ('cur', 'pa', 'pc'):
        ax[key].spines[['top', 'right']].set_visible(False)
    for key, s, t in (('aP', '(a)', 'Concentration fields'), ('cur', '(b)', 'Applied current'),
                      ('pa', '(c)', 'Radial profiles')):
        ax[key].text(-0.13 if key != 'cur' else -0.06, 1.22 if key == 'aP' else 1.10,
                     (r'\textbf{%s} %s' if TEX else '%s %s') % (s, t),
                     transform=ax[key].transAxes, va='bottom', ha='left', fontsize=9)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, STEM + '.pdf'))
    log.info('written %s/%s.pdf: %s %s %+.0f C, %d samples, %.0f min', FIG, STEM, CELL, PROTOCOL, T_C, len(t) - 1,
             tm[-1])


if __name__ == '__main__':
    main()
