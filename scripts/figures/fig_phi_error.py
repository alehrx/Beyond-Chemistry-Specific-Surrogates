"""
fig_phi_error.py
================
Roll-out error against where the condition falls in the training distribution of Phi.

Every one of the 120 roll-outs (5 cells x 4 loads x 3 temperatures x 2 electrodes) is drawn at its
peak flux with the sign of its DISCHARGE phase: anodes on the positive side, cathodes on the
negative side (a cycle visits the opposite sign on charge with the same magnitude). Above it, the
training density Phi ~ N(0, 1.5^2) with the 1 and 2 sigma bands. Error: full-profile RMSE without
volume weighting, mean over the four seeds, window rule tau <= 0.05 or a 5 % current change, 51-point
scoring grid (runs/studies/trained_model/results.csv). Single-column figure.

Writes figures/fig_phi_error.pdf (or MIONET_FIG_DIR).
"""
import os
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm, spearmanr

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
CSV = os.path.join(SM, 'runs', 'studies', 'trained_model', 'results.csv')
FIG = os.environ.get('MIONET_FIG_DIR') or os.path.join(SM, 'figures')        # override: paper folder
MODELS = ['trained_model_s42_lbfgs', 'trained_model_s43_lbfgs', 'trained_model_s44_lbfgs', 'trained_model_s45_lbfgs']
SCORING, METRIC, SIGMA = 'mean-adapt0.05-direct-fv51', 'pct_full', 1.5
SIGN = {'anode': 1.0, 'cathode': -1.0}                  # sign of Phi on discharge (Chemistry.sign)
CELLS = {'Ai2020': ('Ai2020', 'o', '#0072B2'), 'Marquis2019': ('Marquis2019', 's', '#D55E00'),
         'Mohtat2020': ('Mohtat2020', '^', '#009E73'), 'Chen2020': ('Chen2020', 'D', '#E69F00'),
         'Prada2013': ('Prada2013', 'v', '#CC79A7')}
LEGACY = {'LCO 2.3 (Ai)': 'Ai2020', 'LCO 0.6 (Marquis)': 'Marquis2019', 'NMC PC (Mohtat)': 'Mohtat2020',
          'NMC Cyl (Chen)': 'Chen2020', 'LFP Cyl (Prada)': 'Prada2013'}   # names in older CSVs
TEX = mpl.rcParams['text.usetex']
log = logging.getLogger(__name__)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=CSV, help='results file')
    ap.add_argument('--models', nargs='+', default=MODELS, help='the seeds: mean and range over them')
    ap.add_argument('--scoring', default=SCORING, help='ds_mode tag of the rows to plot')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    d = pd.read_csv(a.csv)
    d['cell'] = d['cell'].replace(LEGACY)
    d = d[d.model.isin(a.models) & (d.ds_mode == a.scoring)]
    g = d.groupby(['cell', 'protocol', 'electrode', 'T_C']).agg(
        phi=('phi_max', 'mean'), e=('pct_full', 'mean'), lo=('pct_full', 'min'), hi=('pct_full', 'max')).reset_index()
    g['phi_s'] = g.phi * g.electrode.map(SIGN)

    fig, (ax0, ax) = plt.subplots(2, 1, figsize=(85 * mm, 82 * mm), sharex=True, layout='constrained',
                                  gridspec_kw=dict(height_ratios=[1, 2.6]))
    x = np.linspace(-3.6, 3.6, 500)
    ax0.fill_between(x, norm.pdf(x, scale=SIGMA), color='0.85', lw=0)
    ax0.fill_between(x, norm.pdf(x, scale=SIGMA), where=np.abs(x) <= SIGMA, color='0.70', lw=0)
    ax0.plot(x, norm.pdf(x, scale=SIGMA), color='0.3', lw=0.8)
    ax0.set_ylim(0, None)
    ax0.set_yticks([])
    ax0.set_ylabel('training\ndensity', fontsize=8)
    ax0.spines[['top', 'right', 'left']].set_visible(False)
    for ax_ in (ax0, ax):
        for k in (-2, -1, 1, 2):
            ax_.axvline(k * SIGMA, color='0.6', lw=0.5, ls='--', zorder=1)
        ax_.axvline(0, color='0.85', lw=0.5, zorder=0)
    for k, s in ((-2, r'$-2\sigma$'), (-1, r'$-\sigma$'), (1, r'$\sigma$'), (2, r'$2\sigma$')):
        ax0.text(k * SIGMA, 1.0, s, transform=ax0.get_xaxis_transform(), ha='center', va='bottom', fontsize=7)
    for cell, (lab, mk, col) in CELLS.items():
        s = g[g.cell == cell]
        ax.errorbar(s.phi_s, s.e, yerr=[s.e - s.lo, s.hi - s.e], ls='none', marker=mk, ms=3.4, mfc=col, mec='k',
                    mew=0.3, ecolor=col, elinewidth=0.5, capsize=0, zorder=3, label=lab)
    ax.set_yscale('log')
    ax.set_xlim(-3.6, 3.6)
    ax.set_xlabel(r'peak flux $\Phi$' if TEX else 'peak flux Phi')
    ax.set_ylabel('RMSE [%s]' % (r'\%' if TEX else '%'))
    fig.suptitle('Roll-out error against the dimensionless flux', fontsize=9, x=0.02, ha='left')
    ax.text(0.98, 0.97, 'anodes', transform=ax.transAxes, ha='right', va='top',
            fontsize=7, color='0.35')
    ax.text(0.02, 0.97, 'cathodes', transform=ax.transAxes, ha='left', va='top',
            fontsize=7, color='0.35')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=6.5, loc='lower left', ncol=1, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, 'fig_phi_error.pdf'))
    log.info('written %s/fig_phi_error.pdf | %d roll-outs, Spearman |Phi|~error %.2f, max |Phi| = %.2f sigma',
             FIG, len(g), spearmanr(g.phi, g.e).statistic, g.phi.max() / SIGMA)


if __name__ == '__main__':
    main()
