"""
fig_training_data.py
====================
Training-set figure for the dimensionless formulation, drawn from the ACTUAL sampler used in
training (srcs.prepData), configured from the reported run's own config.json.

    python scripts/figures/fig_training_data.py                          # the reported run (trained_model_s42)
    python scripts/figures/fig_training_data.py --run trained_model_s43  # any other run

  (a) initial profiles x0 = s GRF(xi) - Phi_prev xi^2/2: each drawn profile (solid, read at the
      sensors) with its parabolic component (dashed), so the two ingredients stay visible. The
      parabola is the tau -> infinity solution under a constant flux, the shape a particle
      relaxes to under a sustained current. Profiles are shown at their sampled amplitude.
  (b) the distribution of the dimensionless flux, on the normalised axis Phi / sigma_Phi (the
      value of sigma_Phi is given in the results, not here).
  (c) the collocation points of every loss term - PDE, initial condition, the two boundary
      conditions, and the radial quadrature of the lithium balance at its fixed times - with the
      marginal densities of tau (below) and xi (right) measured on a large draw.

The number of sensors is deliberately not printed: it is introduced in the results section.
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm

mpl.rcParams.update({
    'font.family': 'serif', 'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'axes.prop_cycle': mpl.cycler('color', ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7', '#56B4E9']),
    'lines.linewidth': 1.2, 'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.minor.visible': True, 'ytick.minor.visible': True, 'xtick.major.size': 4, 'ytick.major.size': 4,
    'xtick.minor.size': 2, 'ytick.minor.size': 2, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6, 'lines.markersize': 4,
    'axes.xmargin': 0.02, 'axes.ymargin': 0.05, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.dpi': 300,
    **({'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
        'pgf.texsystem': 'pdflatex', 'pgf.rcfonts': False}
       if __import__('shutil').which('latex') else {'text.usetex': False, 'mathtext.fontset': 'cm'}),
})
mm = 1 / 25.4

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))                # the repository root
FIG = os.environ.get('MIONET_FIG_DIR') or os.path.join(ROOT, 'figures')       # override: paper folder
sys.path.insert(0, ROOT)
logging.basicConfig(level=logging.WARNING)
from srcs.prepData import Config, data                      # noqa: E402

log = logging.getLogger(__name__)

ap = argparse.ArgumentParser()
ap.add_argument('--run', default='trained_model_s42', help='run whose sampler configuration is drawn')
ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--suffix', default='')
A = ap.parse_args()

C_PDE, C_IC, C_BC, C_MASS = '0.45', '#D55E00', '#0072B2', '#009E73'
N_PDE_SHOW, N_IC_SHOW, N_BC_SHOW = 280, 14, 30     # points drawn in (c); the training set is far denser
N_MASS_SHOW = 15                                    # quadrature nodes drawn per balance time

# ---- the sampler exactly as the run configured it ----
run_dir = os.path.join(ROOT, 'runs', 'v2_nondim', A.run)
cfg_run = json.load(open(os.path.join(run_dir, 'config.json')))
cd = dict(cfg_run['config_dict'])
cd.update(N_pde=1, Ny_pde=4000, N_ic=1, Ny_ic=N_IC_SHOW, N_bc=1, Ny_bc=4000, num_batches=1,
          device='cpu')
known = set(Config.__dataclass_fields__)
D = data(Config(**{k: v for k, v in cd.items() if k in known}))
sensors = D.sensors1
cons_t = int(cfg_run.get('v2', {}).get('cons_n_t', 4))
np.random.seed(A.seed)

# ------------------------------------------------------------------ (a) input profiles
# Drawn by the sampler itself; the parabolic part is separated through the returned parts.
# A fixed mix of branches so the panel shows a matched, an independent and a parabola-free case.
# The GRF is a perturbation of the parabola (its amplitude is a fraction of the parabola's), so
# examples are picked with a strong GRF weight s and a moderate Phi_prev, of both signs, for the
# two ingredients to be told apart by eye. They are real draws, only selected.
want = [('matched', 1), ('matched', -1), ('independent', 1), ('independent', -1), ('grf', 0), ('grf', 0)]
profiles = []
while want:
    phi = float(np.random.normal(0.0, 1.0) * D.SF2)
    _, parts = D._profile(phi, return_parts=True)
    br, pp, gs_ = parts['branch'], float(parts['phi_prev']), float(parts['grf_scale'])
    key = (br, int(np.sign(pp)))
    if key in want and gs_ > 0.4 and (br == 'grf' or 0.1 < abs(pp) < 1.0):
        want.remove(key)
        profiles.append((br, pp, D.xi_fine.copy(), parts['fine'].copy()))
cols = ['#0072B2', '#56B4E9', '#D55E00', '#E69F00', '#CC79A7', '#009E73']

# ------------------------------------------------------------------ (b) flux distribution
phi_draw = np.array([D._draw_phi(np.random.uniform(D.SL2_min, D.SL2_max))[0]
                     for _ in range(20000)]) / D.SF2

# ------------------------------------------------------------------ (c) collocation points
X_pde = D.data_pde(1, 4000)[0, :, :2].numpy()                 # columns: tau/tau_max, xi
X_bc = D.data_bc(1, 4000)[0, :, :1].numpy().ravel()            # tau of the BC points
X_ic = D.data_ic(1, N_IC_SHOW)[0, :, 1].numpy().ravel()        # xi of the IC points
rng = np.random.default_rng(A.seed)
pde_show = X_pde[rng.choice(len(X_pde), N_PDE_SHOW, replace=False)]
# BC points for display: evenly spaced QUANTILES of the BC tau density rather than a random
# handful, which with ~30 points clumps by chance. Same distribution, no clumping.
bc_show = np.quantile(X_bc, (np.arange(N_BC_SHOW) + 0.5) / N_BC_SHOW)
mass_t = np.arange(1, cons_t + 1) / cons_t
mass_xi = np.linspace(0.0, 1.0, N_MASS_SHOW)

# ------------------------------------------------------------------ figure
# The two marginals of (c) are attached to its axes as insets rather than nested in the layout:
# constrained_layout cannot size a nested grid with shared axes and an outside legend, and
# collapsed panel (c) to a sliver. Insets are placed in axes fraction and saved with bbox='tight'.
fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(170 * mm, 62 * mm), layout='constrained',
                                       gridspec_kw=dict(width_ratios=[1.0, 1.0, 0.85]))
ax_t = ax_c.inset_axes([0.0, -0.52, 1.0, 0.24])          # p(tau), below, hanging downwards
ax_x = ax_c.inset_axes([1.05, 0.0, 0.26, 1.0])           # p(xi), right

# (a)
# shown on a common scale: every profile multiplied by ONE factor so the largest |x0| of the set
# is AMP_SHOW (the operator is linear in x0, so only the shape is informative; relative
# amplitudes between the drawn profiles are kept)
AMP_SHOW = 0.25
_k = AMP_SHOW / max(float(np.abs(pr).max()) for *_, pr in profiles)
profiles = [(br, pp, xi, _k * pr) for br, pp, xi, pr in profiles]
for (br, pp, xi, prof), c in zip(profiles, cols):
    ax_a.plot(xi, prof, '-', color=c, lw=1.2)
    ax_a.plot(sensors, np.interp(sensors, xi, prof), 'o', color=c, ms=2.4, lw=0)
ax_a.axhline(0, color='0.85', lw=0.6, zorder=0)
ax_a.plot([], [], '-', color='0.35', lw=1.2, label=r'$x_0(r)$')
ax_a.plot([], [], 'o', color='0.35', ms=2.4, lw=0, label='sensors')
# headroom above the curves so the legend never sits on a profile, whatever the draw
_lo = min(float(pr.min()) for *_, pr in profiles)
_hi = max(float(pr.max()) for *_, pr in profiles)
ax_a.set_ylim(-1.2 * AMP_SHOW, 1.45 * AMP_SHOW)
ax_a.legend(loc='upper left', fontsize=6.5, handlelength=1.6, handletextpad=0.4, labelspacing=0.25,
            borderaxespad=0.2)
ax_a.set_xlabel(r'radius $r$')
ax_a.set_ylabel(r'$x_0(r)$, centre-pinned', labelpad=2)
ax_a.set_title('(a) input profiles', fontsize=8.5)

# (b)
g = np.linspace(-4, 4, 400)
# shaded one- and two-sigma bands; the shares are those of the normal distribution drawn
p1, p2 = 100 * (norm.cdf(1) - norm.cdf(-1)), 100 * (norm.cdf(2) - norm.cdf(-2))
pct = r'\,\%' if mpl.rcParams['text.usetex'] else r'%'
g2, g1 = g[np.abs(g) <= 2], g[np.abs(g) <= 1]
ax_b.plot(g, norm.pdf(g), 'k-', lw=1.3, zorder=2, label=r'$\mathcal{N}(0,1)$')
ax_b.fill_between(g2, norm.pdf(g2), color='0.92', lw=0, zorder=0,
                  label=r'$\pm2\sigma_\Phi$: %.1f%s' % (p2, pct))
ax_b.fill_between(g1, norm.pdf(g1), color='0.78', lw=0, zorder=1,
                  label=r'$\pm1\sigma_\Phi$: %.1f%s' % (p1, pct))
ax_b.set_xlim(-4, 4)
ax_b.set_ylim(0, norm.pdf(0) * 1.45)   # room for the legend above the peak
ax_b.set_xlabel(r'normalised flux $\Phi/\sigma_\Phi$')
ax_b.set_ylabel('probability density')
ax_b.legend(loc='upper left', fontsize=5.5, handlelength=1.1, handleheight=0.6, handletextpad=0.3,
            labelspacing=0.2, borderaxespad=0.2)
ax_b.set_title('(b) flux distribution', fontsize=8.5)

# (c) main
ax_c.fill_between([0, 1], [1, 1], color='0.98', zorder=0)
for k, tm in enumerate(mass_t):
    ax_c.plot([tm, tm], [0, 1], '-', color=C_MASS, lw=0.6, alpha=0.6, zorder=1)
    ax_c.plot(np.full_like(mass_xi, tm), mass_xi, 'o', color=C_MASS, ms=2.0, zorder=2,
              label='LB' if k == 0 else None)
ax_c.plot(pde_show[:, 0], pde_show[:, 1], 'o', ms=1.3, color=C_PDE, label='PDE', zorder=3)
ax_c.plot(np.zeros_like(X_ic), X_ic, 'o', ms=2.6, color=C_IC, label='IC', zorder=4, clip_on=False)
ax_c.plot(bc_show, np.ones_like(bc_show), 'o', ms=2.6, color=C_BC, label='BC', zorder=4, clip_on=False)
ax_c.plot(bc_show, np.zeros_like(bc_show), 'o', ms=2.6, color=C_BC, zorder=4, clip_on=False)
ax_c.set_xlim(0, 1)
ax_c.set_ylim(0, 1)
ax_c.set_ylabel(r'radius $r$')
ax_c.set_title('(c) collocation points', fontsize=8.5)
ax_c.set_xlabel(r'time $\tau/\tau_{\max}$', labelpad=1)    # tau ticks are read on the main panel

# (c) marginals, measured on the full draw
bins = np.linspace(0, 1, 21)
ax_t.hist(X_pde[:, 0], bins=bins, density=True, color='0.88', edgecolor='white', linewidth=0.4)
ax_t.plot([0, 1], [2, 0], 'k-', lw=1.0)
ax_t.set_ylim(0, 2.0)                                     # axis ends where the density does
ax_t.invert_yaxis()                                       # the triangle hangs from the main panel
ax_t.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax_t.set_ylabel(r'$p(\tau)$', fontsize=7.5)
ax_t.tick_params(axis='y', labelsize=6.5)
ax_x.hist(X_pde[:, 1], bins=bins, density=True, orientation='horizontal', color='0.88',
          edgecolor='white', linewidth=0.4)
ax_x.plot([1, 1], [0, 1], 'k-', lw=1.0)
ax_x.set_xlim(0, 1.0)                                     # axis ends where the density does
ax_x.set_xticks([1.0])                                    # a '0' here would collide with tau = 1
ax_x.set_xlabel(r'$p(r)$', fontsize=7.5)
ax_x.tick_params(axis='x', labelsize=6.5)
ax_x.set_ylim(0, 1)
ax_t.set_xlim(0, 1)
plt.setp(ax_x.get_yticklabels(), visible=False)
_h, _l = ax_c.get_legend_handles_labels()
_order = sorted(range(len(_l)), key=lambda i: _l[i] == 'LB')      # LB last, the rest as drawn
fig.legend([_h[i] for i in _order], [_l[i] for i in _order], loc='outside lower right', ncol=4, fontsize=6.5,
           frameon=False, columnspacing=0.9, handletextpad=0.3)

for a in (ax_a, ax_b, ax_c, ax_t, ax_x):
    a.spines[['top', 'right']].set_visible(False)
ax_t.spines['top'].set_visible(True)                      # inverted: its baseline is now on top
ax_t.spines['bottom'].set_visible(False)
os.makedirs(FIG, exist_ok=True)
fig.savefig(os.path.join(FIG, f'fig_training_data_nd{A.suffix}.pdf'))
print('run %s | sensors %d | xi colloc %s | tau density %s (BC %s) | balance times %s'
      % (A.run, len(sensors), D.xi_colloc, D.tau_density, D.tau_density_bc, mass_t))
print('fig_training_data_nd written')
