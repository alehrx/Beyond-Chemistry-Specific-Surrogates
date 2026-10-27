"""
fig_hpo_v2.py
=============
Hyperparameter-search figure.

    (a) optimisation history: every trial, the running best, and the selected configuration
    (b) fANOVA importance, loss weights separated from architecture and optimiser

Writes figures/fig_hpo.pdf.  Run button; no arguments.

    python scripts/figures/fig_hpo.py                                   # runs/hpo_study (the paper's figure)
    python scripts/figures/fig_hpo.py --run <folder>                    # any other study under runs/
"""
import argparse
import os
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

mpl.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.prop_cycle': mpl.cycler('color', ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7', '#56B4E9']),
    'lines.linewidth': 1.4, 'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.minor.visible': True, 'ytick.minor.visible': True, 'xtick.major.size': 4, 'ytick.major.size': 4,
    'xtick.minor.size': 2, 'ytick.minor.size': 2, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6, 'lines.markersize': 4, 'errorbar.capsize': 3,
    'axes.xmargin': 0.02, 'axes.ymargin': 0.05, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.dpi': 300,
    **({'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
        'pgf.texsystem': 'pdflatex', 'pgf.rcfonts': False}
       if __import__('shutil').which('latex') else {'text.usetex': False, 'mathtext.fontset': 'cm'}),
})
mm = 1 / 25.4
TEX = mpl.rcParams['text.usetex']

PAPER = os.path.dirname(os.path.abspath(__file__))


SM_ROOT = os.path.abspath(os.path.join(PAPER, os.pardir, os.pardir))
N_SEEDS = 20
STUDIES = {                       # run folder -> (study name, objective label)
    'hpo_rollout_s10_p2_noproj': ('MIONet_ND_HPO_rollout', r'roll-out objective (MSE)'),
    'hpo_study': ('MIONet_ND_HPO', r'validation objective $\mathcal{E}$'),
}
FIG_DIR = os.environ.get('MIONET_FIG_DIR') or os.path.join(SM_ROOT, 'figures')   # override: paper folder

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

WEIGHTS = {'w_pde', 'w_ic', 'w_bc', 'w_cons'}
LBL = {'w_cons': r'$\lambda_{\mathrm{MC}}$', 'w_ic': r'$\lambda_{\mathrm{IC}}$',
       'w_pde': r'$\lambda_{\mathrm{PDE}}$', 'w_bc': r'$\lambda_{\mathrm{BC}}$',
       'lr': 'learning rate', 'hidden_dim': 'width',
       'num_hidden_layers': 'depth', 'activation': 'activation'}
C_W, C_A = '#0072B2', '#9C9C9C'          # loss weights / architecture and optimiser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='hpo_study',                      # the paper's HPO figure
                    help='HPO folder: a name under runs/ or a path to a folder holding study.db')
    HPO_RUN = ap.parse_args().run
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folder = HPO_RUN if os.path.isdir(HPO_RUN) else os.path.join(SM_ROOT, 'runs', HPO_RUN)
    storage = 'sqlite:///' + os.path.join(os.path.abspath(folder), 'study.db').replace('\\', '/')
    key = os.path.basename(os.path.normpath(folder))
    HPO_RUN = key
    if key in STUDIES:
        study_name, ylabel = STUDIES[key]
    else:                                         # any other study: the only one in its database
        study_name = optuna.get_all_study_names(storage)[0]
        ylabel = ('roll-out objective (MSE)' if 'rollout' in study_name
                  else r'validation objective $\mathcal{E}$')
    st = optuna.load_study(study_name=study_name, storage=storage)

    done = [t for t in st.trials if t.value is not None and np.isfinite(t.value)]
    n = np.array([t.number for t in done])
    v = np.array([t.value for t in done])
    best = np.minimum.accumulate(v)
    n_pruned = sum(t.state.name == 'PRUNED' for t in st.trials)
    b = st.best_trial

    runs = {}
    for seed in range(N_SEEDS):
        ev = optuna.importance.FanovaImportanceEvaluator(seed=seed)
        for k, val in optuna.importance.get_param_importances(st, evaluator=ev).items():
            runs.setdefault(k, []).append(val)
    imp = dict(sorted({k: float(np.mean(x)) for k, x in runs.items()}.items(), key=lambda kv: -kv[1]))
    sd = {k: float(np.std(x)) for k, x in runs.items()}

    fig = plt.figure(figsize=(170 * mm, 66 * mm), layout='constrained')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])

    # ── (a) optimisation history ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(n, v, 'o', ms=3.6, mfc='none', mew=0.9, color='0.45', label='trial')
    ax.semilogy(n, best, '-', color='#D55E00', lw=1.6, label='running best')
    ax.semilogy([b.number], [b.value], '*', ms=13, color='#D55E00', mec='k', mew=0.5,
                label='selected', zorder=5)
    ax.set_xlabel('trial')
    ax.set_ylabel(ylabel)
    ax.set_title((r'\textbf{(a)} Search history' if TEX else '(a) Search history'), fontsize=9, loc='left')
    ax.legend(loc='upper right', markerscale=0.55, handletextpad=0.5, labelspacing=0.35)
    ax.text(0.97, 0.63, r'%d trials, %d pruned' % (len(st.trials), n_pruned),
            transform=ax.transAxes, ha='right', fontsize=7.5, color='0.35')
    ax.spines[['top', 'right']].set_visible(False)

    # ── (b) importance, weights against the rest ────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    keys = list(imp)[::-1]
    cols = [C_W if k in WEIGHTS else C_A for k in keys]
    ax.barh(range(len(keys)), [imp[k] for k in keys], color=cols, height=0.72,
            xerr=[sd[k] for k in keys], error_kw=dict(ecolor='0.3', lw=0.9, capsize=2.5))
    ax.set_yticks(range(len(keys)), [LBL.get(k, k) for k in keys])
    ax.set_xlabel('fANOVA importance')
    ax.set_title((r'\textbf{(b)} Hyperparameter importance' if TEX else '(b) Hyperparameter importance'), fontsize=9,
                 loc='left')
    ax.set_xlim(0, max(imp.values()) * 1.55)
    for lab, k in zip(ax.get_yticklabels(), keys):
        lab.set_color(C_W if k in WEIGHTS else '0.35')
    ax.legend(handles=[Line2D([], [], lw=5, color=C_W, label='loss weights'),
                       Line2D([], [], lw=5, color=C_A, label='architecture')],
              loc='lower right', handlelength=1.1, handletextpad=0.5, labelspacing=0.35,
              borderaxespad=0.4)
    ax.spines[['top', 'right']].set_visible(False)

    for ext in ('.pdf',):
        fig.savefig(os.path.join(FIG_DIR, 'fig_hpo' + ext))
    log.info('fANOVA over %d seeds:', N_SEEDS)
    for k in imp:
        log.info('   %-20s %.3f +- %.3f', k, imp[k], sd[k])
    log.info('best trial %d: E = %.3e, lambda_M = %.3g', b.number, b.value, b.params['w_cons'])
    log.info('written figures/fig_hpo.pdf')


if __name__ == '__main__':
    main()
