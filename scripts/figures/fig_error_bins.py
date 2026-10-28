"""
fig_error_bins.py
=================
The accuracy of the surrogate over every cell and protocol, as two figures.

    python scripts/figures/fig_error_bins.py                  # averaged over temperature
    python scripts/figures/fig_error_bins.py --by-temperature # one column of panels per temperature
    python scripts/figures/fig_error_bins.py --metric pct_full_vol

FIGURE 1 (fig_results_bins)  two stacked panels, anode above cathode, one group of bars per cell
and one bar per protocol within the group, averaged over the isothermal runs. The bar is the mean
over the cross-validation models and the whisker spans their min-max, so the dispersion of the
TRAINING PROCEDURE is visible rather than the score of one lucky network. The whisker is omitted
when only one model is available. Miniatures of the two current profiles sit below.

FIGURE 2 (fig_results_bins_byT)  the same bars, but one column of panels per temperature, so the
temperature dependence that figure 1 averages away can be read directly.

The temperature grid is 0 / 25 / 50 C.

Colours follow the rest of the paper: blue for the anode, orange for the cathode, with protocol
shown by shade within the electrode's hue.

Reads runs/studies/mionet_77k/results.csv (srcs/evaluation.py, the scores of the four networks).
"""
import argparse
import os
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

mpl.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.prop_cycle': mpl.cycler('color', ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7', '#56B4E9']),
    'lines.linewidth': 1.2, 'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'xtick.major.size': 4, 'ytick.major.size': 4, 'xtick.minor.size': 2, 'ytick.minor.size': 2,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
    'lines.markersize': 4, 'errorbar.capsize': 2,
    'axes.xmargin': 0.02, 'axes.ymargin': 0.05, 'legend.frameon': False,
    'savefig.bbox': 'tight', 'savefig.dpi': 300,
    **({'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
        'pgf.texsystem': 'pdflatex', 'pgf.rcfonts': False}
       if __import__('shutil').which('latex') else {'text.usetex': False, 'mathtext.fontset': 'cm'}),
})
mm = 1 / 25.4
BOX_LABEL = 'seed quartiles'
SHOW_BOX = True     # --no-box: whiskers (min-max over the seeds) only
YLIM = None         # --ylim: one common y-range for every panel, anode and cathode alike

PAPER = os.path.dirname(os.path.abspath(__file__))
SM = os.path.abspath(os.path.join(PAPER, os.pardir, os.pardir))   # the repository root
sys.path.insert(0, SM)
sys.path.insert(0, os.path.join(SM, 'srcs'))

log = logging.getLogger(__name__)
CSV = [os.path.join(SM, 'runs', 'studies', 'mionet_77k', 'results.csv')]   # --csv: the paper's scores
FIG = os.environ.get('MIONET_FIG_DIR') or os.path.join(SM, 'figures')

CELL_ORDER = ['Ai2020', 'Marquis2019', 'Mohtat2020', 'Chen2020', 'Prada2013']   # the paper's names
CELL_LABELS = {c: c for c in CELL_ORDER}
LEGACY = {'LCO 2.3 (Ai)': 'Ai2020', 'LCO 0.6 (Marquis)': 'Marquis2019', 'NMC PC (Mohtat)': 'Mohtat2020',
          'NMC Cyl (Chen)': 'Chen2020', 'LFP Cyl (Prada)': 'Prada2013'}   # names in older CSVs
PROTOCOLS = ['CC-CV 0.5C', 'CC-CV 1C', 'CC-CV 3C', 'DST 3C']
SHADES = {'anode': ['#9ecae1', '#6baed6', '#3182bd', '#08519c'],
          'cathode': ['#fdbe85', '#fd8d3c', '#e6550d', '#a63603']}
GREYS = ['0.82', '0.62', '0.42', '0.20']
# One short label for every metric. The long per-metric strings overflowed the panel in the
# by-temperature layout, where the two rows' labels overprinted each other into illegible mush.
# Which metric a figure shows is carried by its filename suffix and its caption, not its axis.
TEX = mpl.rcParams['text.usetex']
YLABEL = 'RMSE [\\%]' if TEX else 'RMSE [%]'


def metric_ylabel(metric):
    """State the denominator explicitly; a bare percent is ambiguous here."""
    if metric.endswith('_cmax'):
        return ('RMSE (\\% of $c_{s,\\max}$)' if mpl.rcParams['text.usetex']
                else r'RMSE (% of $c_{s,max}$)')
    if metric.endswith('_max'):
        return ('RMSE (\\% of maximum stoichiometry)' if mpl.rcParams['text.usetex']
                else 'RMSE (% of maximum stoichiometry)')
    return 'RMSE [\\%]' if mpl.rcParams['text.usetex'] else 'RMSE [%]'


def cv_models(d, suffix='', n_seeds=5):
    """The cross-validation family, by EXACT name.

    Prefix matching was a trap: once the seeds have L-BFGS-refined counterparts,
    startswith('particle_cv_uniform') matches particle_cv_uniform_s0, _s0_lbfgs AND
    _s0_lbfgs/best_pi alike, so the box would be built from a blend of refined and un-refined
    models - fifteen where there should be five - and the figure would be quietly meaningless.
    `suffix` selects which generation to plot: '' for the originals, '_lbfgs/best_pi' for the
    refined ones."""
    have = set(d.model)
    for tag in ('particle_cv_uniform', 'particle_cv_surface'):
        m = ['%s_s%d%s' % (tag, k, suffix) for k in range(n_seeds)]
        m = [x for x in m if x in have]
        if len(m) >= 2:
            if len(m) < n_seeds:
                log.warning('only %d of %d seeds present for suffix %r: %s',
                            len(m), n_seeds, suffix, ', '.join(m))
            return m, True
    if suffix:
        raise SystemExit('no cross-validation family found with suffix %r - has the refinement '
                         'run and been scored?' % suffix)
    pref = [m for m in ('particle_iso_uniform', 'particle_pi_strat') if m in have]
    return (pref[:1] or sorted(have)[:1]), False


def aggregate(d, metric, by_temperature):
    """Distribution over the cross-validation seeds. Temperature is averaged first WITHIN each
    model, so the spread measures the training procedure, never the temperature spread.

    Returns mean/min/max/median/q1/q3: the bar is the mean, the box spans the quartiles with the
    median ruled across it, and the whisker spans min to max. With five seeds the quartiles are
    interpolated from five points - the box says where the middle of the procedure's distribution
    sits, not a percentile anyone should read to two digits."""
    keys = ['cell', 'protocol', 'electrode'] + (['T_C'] if by_temperature else [])
    per_model = d.groupby(keys + ['model'])[metric].mean().reset_index()
    return per_model.groupby(keys)[metric].agg(
        mean='mean', min='min', max='max', count='count', median='median',
        q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75)).reset_index()


def model_series(d_all, model, metric, by_temperature):
    """One named model at the same points, averaged the same way (the --extra-model dash)."""
    if not model or model == 'none':
        return None
    s = d_all[d_all.model == model]
    if s.empty:
        log.warning('model %r not in results.csv - skipping its series', model)
        return None
    keys = ['cell', 'protocol', 'electrode'] + (['T_C'] if by_temperature else [])
    return s.groupby(keys)[metric].mean().reset_index().rename(columns={metric: 'value'})


EX_C = '#CC79A7'          # a third hue: a single named model drawn against the seed band


def draw_bars(ax, g, cells, el, is_cv, x, width, shades, extra=None):
    for j, p in enumerate(PROTOCOLS):
        sub = g[(g.electrode == el) & (g.protocol == p)].set_index('cell').reindex(cells)
        pos = x - 0.4 + width * (j + 0.5)
        vals = sub['mean'].values
        ax.bar(pos, vals, width * 0.92, color=shades[j], edgecolor='white', linewidth=0.4, zorder=2)
        if is_cv and (sub['count'] > 1).any():
            lo = np.nan_to_num(vals - sub['min'].values)
            hi = np.nan_to_num(sub['max'].values - vals)
            ax.errorbar(pos, vals, yerr=np.vstack([lo, hi]), fmt='none',
                        ecolor='0.25', elinewidth=0.7, capsize=1.8, zorder=3)
            # the seed BOX on top of the bar: quartiles, with the median ruled across
            # (skipped with --no-box: the whiskers above then carry the seed spread alone)
            q1, q3, med = ((sub['q1'].values, sub['q3'].values, sub['median'].values)
                           if SHOW_BOX else ([], [], []))
            bw = width * 0.44
            for xc, a_, b_, m_ in zip(pos, q1, q3, med):
                if not np.isfinite(a_) or not np.isfinite(b_):
                    continue
                ax.add_patch(plt.Rectangle((xc - bw / 2, a_), bw, max(b_ - a_, 0),
                                           facecolor='white', alpha=0.75, edgecolor='0.15',
                                           linewidth=0.6, zorder=4))
                ax.plot([xc - bw / 2, xc + bw / 2], [m_, m_], color='0.1', lw=0.9, zorder=5,
                        solid_capstyle='butt')
        if extra is not None:
            # hlines, not a '_' marker: a marker is sized in POINTS, so the same call drew a dash
            # three times too wide in the narrow by-temperature panels, spilling over its neighbours
            e = extra[(extra.electrode == el) & (extra.protocol == p)].set_index('cell').reindex(cells)
            ax.hlines(e['value'].values, pos - width * 0.46, pos + width * 0.46,
                      color=EX_C, lw=1.3, zorder=7)
    ax.grid(axis='y', color='0.9', lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks(x)
    ax.set_xlim(-0.55, len(cells) - 0.45)
    if YLIM:
        ax.set_ylim(0.0, YLIM)


def legend_handles(is_cv, extra=None, extra_name='', box_name='seed quartiles'):
    from matplotlib.lines import Line2D
    h = [Patch(facecolor=GREYS[j], edgecolor='white', label=p) for j, p in enumerate(PROTOCOLS)]
    if is_cv and SHOW_BOX:
        h.append(Patch(facecolor='white', edgecolor='0.15', linewidth=0.6,
                       label=box_name))
    elif is_cv:
        h.append(Line2D([], [], ls='none', marker='|', ms=9, mew=1.0, color='0.25',
                        label=box_name))
    if extra is not None:
        h.append(Line2D([], [], ls='none', marker='_', ms=9, mew=1.3, color=EX_C, label=extra_name))
    return h


def current_miniatures(ax1, ax2, panel_labels=None):
    # Keep this miniature independent of PyBaMM so figure generation does not
    # import the full results harness merely to draw a fixed standard profile.
    frac = np.ones(361)
    for k, value in ((0, 0), (17, .125), (45, .25), (57, -.125), (65, 0),
                     (81, .125), (105, .25), (117, -.125), (125, 0),
                     (141, .125), (165, .25), (177, -.125), (185, 0),
                     (201, .125), (237, 1), (245, .625), (269, -.25),
                     (277, .25), (309, -.429), (317, 0)):
        frac[k:] = value
    ax2.plot(np.arange(len(frac)) / 60.0, frac, color='k', lw=1.0)
    ax2.axhline(0, color='0.75', lw=0.6, zorder=0)
    ax2.set_xlabel('time [min]')
    # CC discharge (positive, as in every current panel of the paper), CC charge, CV hold
    tt = np.concatenate([np.linspace(0, 1, 60), np.linspace(1, 2, 60), np.linspace(2, 2.8, 40)])
    ii = np.concatenate([np.ones(60), -np.ones(60), -np.exp(-4 * np.linspace(0, 1, 40))])
    ax1.plot(tt, ii, color='k', lw=1.0)
    ax1.axhline(0, color='0.75', lw=0.6, zorder=0)
    ax1.set_xlabel('time [-]')
    ax1.set_xticks([])
    titles = ('CC-CV profile: CC discharge, CC-CV charge', 'DST profile, one repetition (peak 3C)')
    for a, t, label in zip((ax1, ax2), titles, panel_labels or ('', '')):
        a.set_ylabel(r'$I$ [-]' if TEX else 'I [-]', labelpad=1)
        a.set_ylim(-1.1, 1.1)                    # 0.1 between the curve and the axes
        a.set_yticks([-1, 0, 1])
        a.spines[['top', 'right']].set_visible(False)
        a.minorticks_off()
        a.set_title((r'\textbf{%s} %s' if TEX else '%s %s') % (label, t) if label else t,
                    loc='left', fontsize=9, pad=3)


def fig_averaged(d, cells, metric, is_cv, extra=None, extra_name=''):
    g = aggregate(d, metric, by_temperature=False)
    W, H = 170, 126
    fig = plt.figure(figsize=(W * mm, H * mm), layout='constrained')
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.42], hspace=0.05, wspace=0.12)
    ax_a, ax_c = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])
    x, width = np.arange(len(cells)), 0.8 / len(PROTOCOLS)
    for ax, el in ((ax_a, 'anode'), (ax_c, 'cathode')):
        draw_bars(ax, g, cells, el, is_cv, x, width, SHADES[el], extra)
        ax.text(0.012, 0.92, el, transform=ax.transAxes, fontsize=9, va='top', ha='left',
                color=SHADES[el][3])
    fig.supylabel(YLABEL, fontsize=8.5)
    plt.setp(ax_a.get_xticklabels(), visible=False)
    ax_c.set_xticklabels(cells, fontsize=7.5)
    # A FIGURE legend, not an axes one. Anchored to ax_a it makes constrained_layout reserve the
    # width INSIDE that axes, and with the extra entries the panels shrank until the five cell
    # names overprinted each other into one illegible line.
    fig.legend(handles=legend_handles(is_cv, extra, extra_name, box_name=BOX_LABEL),
               loc='outside upper center', ncol=4, fontsize=7.5,
               columnspacing=1.6, handlelength=1.4, handletextpad=0.5)
    current_miniatures(fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]))
    return fig, g


def fig_by_temperature(d, cells, metric, is_cv, extra=None, extra_name=''):
    g = aggregate(d, metric, by_temperature=True)
    temps = sorted(g.T_C.unique())
    # Use one scale for both electrodes and all temperatures. Include the seed
    # whiskers in the recorded maximum, then leave 20% headroom above it.
    common_ylim = float(YLIM) if YLIM else 1.2 * float(g['max'].max())
    log.info('common y-limit: %.3f %% (120%% of max recorded seed error %.3f %%)',
             common_ylim, float(g['max'].max()))
    # Give each temperature column substantially more horizontal room.  The
    # long parameter-set labels otherwise make constrained_layout reserve a
    # disproportionate amount of the available width as inter-panel padding.
    W, H = 170, 140
    fig = plt.figure(figsize=(W * mm, H * mm), layout='constrained')
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, wspace=0.0, hspace=0.02)
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 0.55], hspace=0.08, wspace=0.0)
    axs = np.empty((2, len(temps)), dtype=object)
    x, width = np.arange(len(cells)), 0.8 / len(PROTOCOLS)
    for c, T in enumerate(temps):
        sub_T = g[g.T_C == T]
        ex_T = None if extra is None else extra[extra.T_C == T]
        for r, el in enumerate(('anode', 'cathode')):
            sharex = axs[0, 0] if (r, c) != (0, 0) else None
            sharey = axs[r, 0] if c else None
            ax = axs[r, c] = fig.add_subplot(gs[r, 2*c:2*c+2], sharex=sharex, sharey=sharey)
            draw_bars(ax, sub_T, cells, el, is_cv, x, width, SHADES[el], ex_T)
            if r == 0:
                ax.set_title(r'$%+d\,^{\circ}$C' % int(T), fontsize=9, pad=3, loc='left')
            if c == 0:
                ax.text(0.03, 0.92, el, transform=ax.transAxes, fontsize=9, va='top', ha='left',
                        color=SHADES[el][3])
            else:
                # The rows share one scale; repeating numeric labels in every
                # temperature column wastes width and forces the panels apart.
                ax.tick_params(axis='y', which='both', labelleft=False)
            if r == 1:
                ax.set_xticklabels([CELL_LABELS.get(c_, c_) for c_ in cells], fontsize=7.5, rotation=35,
                                   ha='right')
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_ylim(0.0, common_ylim)
    fig.suptitle((r'\textbf{(a)} %s' if TEX else '(a) %s')
                 % 'Full-profile RMSE of the analysed scenarios', x=0.01, ha='left', fontsize=9)
    # a FIGURE legend, not an axes one: anchoring it to axs[0,0] makes constrained_layout reserve
    # the space inside that column and tears a gap between the first and second panel
    handles = [h for h in legend_handles(is_cv, extra, extra_name, box_name=BOX_LABEL)
               if h.get_label() != BOX_LABEL]
    fig.legend(handles=handles,
               loc='outside upper right', ncol=len(PROTOCOLS) + 3, fontsize=8,
               columnspacing=1.6, handlelength=1.4, handletextpad=0.5,
               )
    current_axes = (fig.add_subplot(gs[2, :3]), fig.add_subplot(gs[2, 3:]))
    current_miniatures(*current_axes, panel_labels=('(b)', '(c)'))
    for r in range(2):                                   # one label per electrode row
        axs[r, 0].set_ylabel(YLABEL, fontsize=9)
    return fig, g


def main():
    global PROTOCOLS
    ap = argparse.ArgumentParser()
    ap.add_argument('--metric', default='pct_surf',
                    choices=['pct_surf', 'pct_full', 'pct_full_vol', 'pct_soc',
                             'pct_surf_max', 'pct_full_max', 'pct_full_vol_max', 'pct_soc_max',
                             'pct_surf_cmax', 'pct_full_cmax', 'pct_full_vol_cmax', 'pct_soc_cmax'],
                    help="pct_full is the uniform average over the radial grid, and is what the "
                         "paper reports. pct_full_vol weights each node by the shell it stands "
                         "for; it is still computed in results.csv but is no longer the default "
                         "full-profile figure")
    ap.add_argument('--by-temperature', action='store_true')
    ap.add_argument('--csv', nargs='+', default=None,
                    help='results file(s), concatenated (default runs/studies/mionet_77k/results.csv)')
    ap.add_argument('--both', action='store_true', help='write both figures (default)')
    ap.add_argument('--protocols', nargs='*', default=None,
                    help='which protocols to show (default %s)' % PROTOCOLS)
    ap.add_argument('--cv-models', nargs='*', default=None,
                    help='build the box from these EXACT model names instead of the '
                         'particle_cv_* family. Use for a family whose seeds are not named '
                         '_s0.._s4, e.g. --cv-models m21_lbfgs m21_s43_lbfgs m21_s44_lbfgs')
    ap.add_argument('--box-label', default=None,
                    help='legend text for the box (default "seed quartiles")')
    ap.add_argument('--cv-suffix', default='',
                    help="which generation of the CV family the box is built from: '' for the "
                         "originals, '_lbfgs/best_pi' for the L-BFGS-refined ones")
    ap.add_argument('--extra-model', default=None,
                    help='a single named model drawn as a second dash on every bar, so one '
                         'network can be read against the seed band it has to beat '
                         "(e.g. particle_iso_uniform_lbfgs/best_pi). 'none' to omit")
    ap.add_argument('--extra-label', default=None, help='legend text for --extra-model')
    ap.add_argument('--suffix', default='', help='appended to the output filenames')
    ap.add_argument('--no-box', action='store_true',
                    help='draw only the min-max whiskers over the seeds, no quartile box')
    ap.add_argument('--ylim', type=float, default=None,
                    help='common upper y-limit [%%] for all panels, anode and cathode alike')
    ap.add_argument('--scoring', default='mean',
                    help="which scoring of the networks to plot, i.e. the ds_mode tag in "
                         "results.csv: 'mean' = one network call per protocol sample, "
                         "'mean-adapt0.03' = the adaptive window rule (srcs/evaluation.py --tau-target)")
    a = ap.parse_args()
    global SHOW_BOX, YLIM, YLABEL
    SHOW_BOX, YLIM = not a.no_box, a.ylim
    YLABEL = metric_ylabel(a.metric)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    global CSV
    if a.csv:
        CSV = [os.path.abspath(c) for c in a.csv]
    for c in CSV:
        if not os.path.exists(c):
            raise SystemExit(f'{c} not found - score the models with srcs/evaluation.py first')

    if a.protocols:
        PROTOCOLS = a.protocols
    d = pd.concat([pd.read_csv(c) for c in CSV], ignore_index=True)
    d['cell'] = d['cell'].replace(LEGACY)
    d = d[d.protocol.isin(PROTOCOLS)]
    if 'ds_mode' in d.columns:
        d = d[d.ds_mode == a.scoring]
    d_all = d[d.cell.isin(CELL_ORDER)]                 # kept whole: the --extra-model rows are in here
    if a.cv_models:
        have = set(d.model)
        missing = [m for m in a.cv_models if m not in have]
        if missing:
            raise SystemExit('not scored: %s' % ', '.join(missing))
        models, is_cv = list(a.cv_models), len(a.cv_models) >= 2
    else:
        models, is_cv = cv_models(d, a.cv_suffix)
    log.info('box over %d model(s): %s', len(models), ', '.join(models))
    global BOX_LABEL
    BOX_LABEL = a.box_label or ('seed quartiles' if SHOW_BOX else 'seed min-max')
    d = d[d.model.isin(models) & d.cell.isin(CELL_ORDER)]
    if 'ds_mode' in d.columns:
        d = d[d.ds_mode == a.scoring]
    cells = [c for c in CELL_ORDER if c in set(d.cell)]
    log.info('models: %s%s', ', '.join(models), '  (cross-validation)' if is_cv else '  (single model)')
    ex_name = a.extra_label or (a.extra_model.split('/')[0].replace('particle_', '')
                                if a.extra_model else '')
    log.info('extra : %s', a.extra_model or '-')
    log.info('cells : %s', ', '.join(cells))
    log.info('temps : %s C', ', '.join('%+.0f' % t for t in sorted(d.T_C.unique())))

    os.makedirs(FIG, exist_ok=True)
    want_both = a.both or not a.by_temperature
    if want_both:
        fig, g = fig_averaged(d, cells, a.metric, is_cv,
                              model_series(d_all, a.extra_model, a.metric, False), ex_name)
        for ext in ('pdf',):
            fig.savefig(os.path.join(FIG, f'fig_results_bins{a.suffix}.{ext}'))
        log.info('\nwritten figures/fig_results_bins.pdf|png  (averaged over temperature)')
        piv = g.pivot_table(index=['cell', 'protocol'], columns='electrode', values='mean')
        denominator = ('c_s,max' if a.metric.endswith('_cmax') else
                       'maximum stoichiometry' if a.metric.endswith('_max') else 'range')
        log.info('\nmean %s [%% of %s], averaged over temperature:\n%s',
                 a.metric, denominator, piv.round(3).to_string())
    if a.by_temperature or a.both:
        fig2, g2 = fig_by_temperature(d, cells, a.metric, is_cv,
                                      model_series(d_all, a.extra_model, a.metric, True), ex_name)
        for ext in ('pdf',):
            fig2.savefig(os.path.join(FIG, f'fig_results_bins_byT{a.suffix}.{ext}'))
        log.info('\nwritten figures/fig_results_bins_byT.pdf|png  (one column per temperature)')
        piv2 = g2.pivot_table(index=['cell', 'protocol'], columns=['electrode', 'T_C'], values='mean')
        denominator = ('c_s,max' if a.metric.endswith('_cmax') else
                       'maximum stoichiometry' if a.metric.endswith('_max') else 'range')
        log.info('\nmean %s [%% of %s] by temperature:\n%s',
                 a.metric, denominator, piv2.round(3).to_string())
    if is_cv:
        spread_source = g if want_both else g2
        spread = (spread_source['max'] - spread_source['min']).describe()
        log.info('\ncross-validation spread (max-min) over %d models: median %.3f, worst %.3f',
                 len(models), spread['50%'], spread['max'])


if __name__ == '__main__':
    main()
