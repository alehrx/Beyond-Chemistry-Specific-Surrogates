"""
fig_training.py
===============
Training history of the reported model (seed 42): the Adam stage followed by the L-BFGS stage, or
the Adam stage alone with --no-lbfgs (the paper figure).

    (a) training loss and held-out physics-informed validation loss
    (b) weighted PDE, IC, BC and lithium-balance terms

Reads runs/v2_nondim/mionet_77k_s42/history.csv (Adam stage) and, without --no-lbfgs,
runs/v2_nondim/mionet_77k_s42_lbfgs/history.csv
(L-BFGS stage). Writes figures/fig_training.pdf.
"""
import json
import os
import logging

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
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
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
ADAM = os.path.join(ROOT, 'runs', 'v2_nondim', 'mionet_77k_s42')
LBFGS = os.path.join(ROOT, 'runs', 'v2_nondim', 'mionet_77k_s42_lbfgs')
FIG = os.environ.get('MIONET_FIG_DIR') or os.path.join(ROOT, 'figures')      # override: paper folder
RESAMPLE = 100
ZOOM_ADAM = 20           # Adam epochs shown before the L-BFGS stage in the zoom of (a)
ADAM_EVERY = 10          # the Adam stage is drawn one epoch in ten: every epoch is too noisy to read
TEX = mpl.rcParams['text.usetex']
log = logging.getLogger(__name__)


def draw(h, n_a, lbfgs=True, best=None):
    """(a) training and held-out physics-informed validation loss; (b) the weighted loss terms.
    lbfgs=False draws the Adam stage alone (h then holds only that stage) and marks the retained
    checkpoint `best` instead of the L-BFGS shading and zoom."""
    fig, axs = plt.subplots(1, 2, figsize=(170 * mm, 65 * mm), layout='constrained')
    # Adam stage thinned to one epoch in ADAM_EVERY (its last epoch always kept); L-BFGS in full
    full = h                                 # the selection markers index the unthinned history
    keep = (h.epoch < n_a) & ((h.epoch % ADAM_EVERY == 0) | (h.epoch == n_a - 1))
    h = h[keep | (h.epoch >= n_a)]
    for ax in axs:
        if lbfgs:
            ax.axvspan(n_a - 0.5, h.epoch.max() + 0.5, color='0.9', lw=0, zorder=0)
        for e in range(RESAMPLE, n_a, RESAMPLE):
            ax.axvline(e, color='0.82', lw=0.5, ls=':', zorder=0)
        ax.set_yscale('log')
        ax.set_xlabel('epoch')
        ax.spines[['top', 'right']].set_visible(False)
        if lbfgs and ax is not axs[0]:                     # (a) labels the stage in its zoom
            ax.text(n_a - 4, 0.97, 'L-BFGS', transform=ax.get_xaxis_transform(), rotation=90,
                    va='top', ha='right', fontsize=7.5)
    kw = dict(lw=0.8)
    ax = axs[0]
    ax.plot(h.epoch, h.train_loss, label='training', **kw)
    ax.plot(h.epoch, h.pi_val_loss, label='validation (held-out)', alpha=0.85, **kw)
    if not lbfgs and best is not None:                     # the retained checkpoint, on the validation curve
        y = float(full.loc[full.epoch == best, 'pi_val_loss'].iloc[0])
        ax.plot([best], [y], '*', ms=7, color='#D55E00', mec='k', mew=0.5, zorder=5,
                label='retained checkpoint')
    ax.set_ylabel('amplitude')
    ax.legend(fontsize=8, loc='lower left' if lbfgs else 'upper right')
    if lbfgs:
        _zoom(ax, full, n_a)
    _terms(axs, h, kw)
    return fig


def _zoom(ax, full, n_a):
    """Inset of (a): the end of the Adam stage and the L-BFGS stage, every epoch drawn."""
    w = full[full.epoch >= n_a - ZOOM_ADAM]
    ins = ax.inset_axes([0.50, 0.50, 0.46, 0.44])
    ins.axvspan(n_a - 0.5, w.epoch.max() + 0.5, color='0.9', lw=0, zorder=0)
    ins.plot(w.epoch, w.train_loss, lw=0.8, marker='o', ms=2)
    ins.plot(w.epoch, w.pi_val_loss, lw=0.8, alpha=0.85, marker='o', ms=2)
    ins.text(n_a + 0.5, 0.45, 'L-BFGS', transform=ins.get_xaxis_transform(), va='center', ha='left', fontsize=7)
    ins.set_yscale('log')
    ins.set_xlim(w.epoch.min(), w.epoch.max() + 0.5)
    ins.tick_params(labelsize=7)
    ins.minorticks_off()
    # the y-range follows the data: fixed decade ticks would stretch the zoom box of a run whose
    # losses span less than a decade far below the curves of the main axes
    lo = float(min(w.train_loss.min(), w.pi_val_loss.min())) / 1.15
    hi = float(max(w.train_loss.max(), w.pi_val_loss.max())) * 1.15
    ins.set_ylim(lo, hi)
    ticks = np.geomspace(lo * 1.1, hi / 1.1, 3)
    ins.set_yticks(ticks, ['%.2g' % v for v in ticks])
    ins.set_title('Zoom on the L-BFGS stage', fontsize=7.5, loc='left', pad=2)
    ax.indicate_inset_zoom(ins, edgecolor='0.4', lw=0.6)


def _terms(axs, h, kw):
    """(b) the weighted loss terms, and the panel titles."""
    ax = axs[1]
    for col, lab in (('pde_loss', 'PDE'), ('ic_loss', 'IC'), ('bc_loss', 'BC'), ('cons_loss', 'MC')):
        ax.plot(h.epoch, h[col], label=lab, **kw)
    ax.set_ylabel('amplitude')
    ax.legend(fontsize=8, ncol=4, loc='upper center', columnspacing=0.8, handlelength=1.5)
    titles = ('Training loss function', 'Independent PI loss terms')
    for ax, l, t in zip(axs, 'ab', titles):
        ax.set_title((r'\textbf{(%s)} %s' if TEX else '(%s) %s') % (l, t), fontsize=9, loc='left')



def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-lbfgs', action='store_true',
                    help='the Adam stage alone, with the retained checkpoint marked (the paper figure)')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    a = pd.read_csv(os.path.join(ADAM, 'history.csv'))
    n_a = len(a)
    best_a = json.load(open(os.path.join(ADAM, 'config.json')))['run_summary']['best_epoch']
    os.makedirs(FIG, exist_ok=True)
    # the Adam stage alone also when the refinement is not there (the published repository ships
    # only the Adam networks)
    if args.no_lbfgs or not os.path.exists(os.path.join(LBFGS, 'history.csv')):
        draw(a, n_a, lbfgs=False, best=best_a).savefig(os.path.join(FIG, 'fig_training.pdf'))
        log.info('Adam %d epochs, best %d (PI val %.3e)', n_a, best_a, a.pi_val_loss.iloc[best_a])
        return
    b = pd.read_csv(os.path.join(LBFGS, 'history.csv'))
    b = b.assign(epoch=b.epoch + n_a)
    h = pd.concat([a, b], ignore_index=True)
    best_b = json.load(open(os.path.join(LBFGS, 'config.json')))['run_summary']['best_epoch']
    draw(h, n_a).savefig(os.path.join(FIG, 'fig_training.pdf'))
    log.info('Adam %d epochs, best %d (PI val %.3e); L-BFGS %d epochs, best %d (PI val %.3e)',
             n_a, best_a, a.pi_val_loss.iloc[best_a], len(b), best_b, b.pi_val_loss.iloc[best_b])


if __name__ == '__main__':
    main()
