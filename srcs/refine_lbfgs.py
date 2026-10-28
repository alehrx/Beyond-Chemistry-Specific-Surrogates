"""refine_lbfgs.py - take a trained MIONet and polish it with L-BFGS on adaptively sampled points.

No architecture change, no new ensemble, no re-tuning: the reported model's exact configuration,
its weights as the starting point, and one second-order phase on collocation points chosen by
where that model's own PDE residual actually is.

    python srcs/refine_lbfgs.py                                   # the reported model
    python srcs/refine_lbfgs.py --from particle_cv_uniform_s3 --lr 3e-4 --epochs 60

WHAT IS DIFFERENT FROM THE ORIGINAL RUN, and only this:
  - INIT_FROM: the weights start where the Adam run finished, so this is a refinement.
  - ADAPTIVE_SAMPLE (pool, k, c): the collocation points are drawn from a pool four times the
    budget and thinned with probability proportional to |r|^k/E[|r|^k] + c. The uniform floor c
    matters - with c = 0 the sampler abandons the regions it currently satisfies and the residual
    simply moves there.
  - ONLY_LBFGS with a low initial step. L-BFGS is a local method on a basin Adam already found;
    the default 1e-3 is a large first step for a converged network, hence 3e-4 here.

Checkpoints are selected on the held-out physics-informed validation loss only.
"""
import os
import sys
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SM_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
PY = sys.executable

DRIVER = '''
import sys, os, json
sys.path.insert(0, %(root)r); sys.path.insert(0, %(scripts)r)
import torch
import train_nd as T

SRC = os.path.join(%(root)r, 'runs', 'v2_nondim', %(src)r)
cfg = json.load(open(os.path.join(SRC, 'config.json')))
cd, v2 = cfg['config_dict'], cfg['v2']

T.RUN_NAME, T.SEED = %(run_name)r, %(seed)d

# ---- the source run's configuration, read from its own artifact ----
T.N_SENSORS = cd['b1_dim']
T.HIDDEN, T.N_LAYERS = cd['hidden_dim'], cd['num_hidden_layers']
T.ACTIVATION = v2['architecture'].get('activation', 'tanh')
T.HARD_BC = bool(v2['architecture'].get('hard_bc', True))   # the source's output form, never the default
T.LR, T.LR_END_FACTOR = cd['lr'], cd['lr_end_factor']
T.LOSS_WEIGHTS = cd['loss_weights']
T.TAU_MAX, T.PHI_SF = v2['tau_max'], v2['phi_sf']
T.PHI_DIST, T.PHI_SPAN = cd['phi_dist'], cd['phi_span']
T.PROFILE_MODE, T.SCALE_FACTOR1, T.SCALE_FACTOR2 = cd['profile_mode'], cd['scale_factor1'], cd['scale_factor2']
T.SCALE_LENGTH1, T.SCALE_LENGTH2 = cd['scale_length1'], cd['scale_length2']
# The adaptive set IS the L-BFGS batch, so it is drawn at the size L-BFGS will use rather than
# at the Adam size and then thinned again: scoring 10x the points and throwing 94%% of them away
# costs an epoch of second derivatives for nothing.
T.N_PDE, T.NY_PDE = cd['N_pde'], %(ny_pde)d
T.N_IC, T.NY_IC = cd['N_ic'], cd['Ny_ic']
T.N_BC, T.NY_BC = cd['N_bc'], cd['Ny_bc']
T.NUM_BATCHES = %(num_batches)s  # the source's value unless --num-batches overrides it
# The three xi laws are pinned to the RESOLVED values the artifact recorded, never left to the
# geometry switch: 'particle' as a GEOMETRY meant the pure equal-volume law when the older runs
# were made and means the hybrid law now, so inheriting it would silently train a different mesh.
# Explicit values beat the preset in prepData - only 'auto'/'follow' fall back to it.
T.XI_MAP, T.XI_COLLOC = v2['xi_map'], v2['xi_colloc']
T.XI_QUAD = v2.get('xi_quad', 'uniform')
T.GEOMETRY = v2.get('geometry', 'uniform' if v2['xi_map'] == 'uniform' else 'particle')
T.GEOMETRY_LAW = v2.get('geometry_law', 'sqrt')
T.PROFILE_SPACE, T.REF_GRID = v2['profile_space'], v2.get('ref_grid', 'fv')
T.CONS_N_KSI, T.CONS_N_T = v2.get('cons_n_ksi', 41), v2.get('cons_n_t', 4)
T.CONS_LEGACY, T.CONS_USE_IC = False, False   # the corrected balance, as everywhere now
T.PDE_FORM = 'conservative'
# the PROFILE ENSEMBLE has to be the one the source was trained on, or the adaptive sampling and
# the L-BFGS phase would be fitting a different input distribution than the weights came from
T.PROFILE_AMP_MAX = cd.get('profile_amp_max', None)
T.PROFILE_LAYER_FRAC = cd.get('profile_layer_frac', 0.0)
T.PROFILE_LAYER_DELTA = cd.get('profile_layer_delta', [0.08, 0.35])
T.PROFILE_AMP_RANDOM = cd.get('profile_amp_random', False)
T.PHI_PREV_MATCH = cd.get('phi_prev_match', 0.5)
T.SHAPED_GRF_FRAC = cd.get('shaped_grf_frac', 0.2)
T.SHAPED_INDEP_FRAC = cd.get('shaped_indep_frac', 0.3)
T.SHAPED_M_RANGE = tuple(cd.get('shaped_m_range', (2.0, 20.0)))
T.SHAPED_NOISE = cd.get('shaped_noise', 0.03)
T.SHAPED_DEPTH_MAX = cd.get('shaped_depth_max', 0.65)
T.ROLLOUT_K = 0
T.TIME_FEATURES = 'linear'
T.T_HI = cd.get('t_hi', 1.0)
# the TIME distribution has to be the source run's too. These three were not carried before
# 2026-10-06, so every refinement until then (m21*_lbfgs, m21d_lbfgs, m51d_lbfgs) drew tau
# UNIFORMLY although its Adam stage had a linearly dropping density - and, for the residual-
# weighted pool, a uniform prior. Runs that predate the keys fall back to the old behaviour.
T.TAU_DENSITY = cd.get('tau_density', None)
T.TAU_DENSITY_BC = bool(cd.get('tau_density_bc', False))
T.TAU_SPLIT = cd.get('tau_split', None)

# ---- the refinement ----
# the CONFIG always comes from the run; the WEIGHTS can come from either of its two
# checkpoints. best_pi/ holds only model.pt and a manifest, so it cannot be the config source.
T.INIT_FROM = os.path.join(SRC, %(init)r) if %(init)r else SRC
T.ONLY_LBFGS = True
T.LBFGS_PHASE = False                 # there is no Adam phase to switch OUT of
T.LR_LBFGS = %(lr)g
T.LBFGS_FULL_BATCH, T.LBFGS_RESAMPLE = True, %(resample)s
T.LBFGS_NY_PDE, T.LBFGS_NY_BC, T.LBFGS_NY_IC = %(ny_pde)d, 200, 100
T.NUM_BATCHES_HINT = 25
T.ADAPTIVE_SAMPLE = %(adaptive)s
T.EPOCHS, T.PATIENCE = %(epochs)d, %(patience)d
T.RESAMPLE_EVERY = 0                  # the adaptive set is the point; do not redraw it uniformly
T.MONITOR, T.MONITOR2 = %(monitor)r, None

print('REFINE %(run_name)s <- %%s | b1=%%d %%dx%%d %%s | tau_max=%%.2f phi_sf=%%.1f | '
      'xi=%%s/%%s/%%s | amp=%%s layer=%%.2f | lr_lbfgs=%%.1e adaptive=%%s ny_pde=%%d | monitor=%%s'
      %% (%(src)r, T.N_SENSORS, T.HIDDEN, T.N_LAYERS, T.ACTIVATION, T.TAU_MAX, T.PHI_SF,
         T.XI_MAP, T.XI_COLLOC, T.XI_QUAD, T.PROFILE_AMP_MAX, T.PROFILE_LAYER_FRAC,
         T.LR_LBFGS, T.ADAPTIVE_SAMPLE, T.LBFGS_NY_PDE, T.MONITOR), flush=True)
T.main()
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='src', default='mionet_77k_s42')
    ap.add_argument('--init', default='model', choices=['model', 'best_pi'],
                    help="which checkpoint to warm-start from: 'model' is the run's model.pt, "
                         "'best_pi' its second checkpoint when it kept one. The configuration is read "
                         "from the run either way")
    ap.add_argument('--tag', default=None)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--pool', type=float, default=4.0, help='candidate pool, x the point budget')
    ap.add_argument('--k', type=float, default=1.0, help='residual exponent in p ~ |r|^k/E+c')
    ap.add_argument('--c', type=float, default=1.0, help='uniform floor in p ~ |r|^k/E+c')
    ap.add_argument('--ny-pde', type=int, default=1000,
                    help='points per profile in the L-BFGS full batch (second derivatives, so this '
                         'is what decides whether it fits in memory)')
    ap.add_argument('--num-batches', default=None,
                    help="'auto' or an int; default: the source run's value. The full batch does not "
                         "depend on it, but 'auto' empties the CUDA cache after the adaptive draw, and a "
                         "fixed count (seed 43 of mionet_77k, originally wk25rcv: 32) left too little memory for the full-batch closure")
    ap.add_argument('--no-adaptive', action='store_true', help='the control: uniform points')
    ap.add_argument('--lbfgs-resample', action='store_true',
                    help='redraw the full-batch subset every epoch (resets the L-BFGS history)')
    ap.add_argument('--monitor', default='pi_val_loss', choices=['pi_val_loss'],
                    help='what early stopping and best.pt follow: the held-out physics loss')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-score', action='store_true',
                    help='accepted for scripts/run_training.py; scoring is its own stage (srcs/evaluation.py)')
    a = ap.parse_args()

    tag = a.tag or ('lbfgs' if a.init == 'model' else 'lbfgs_pi')
    run_name = '%s_%s' % (a.src, tag)
    # one driver file PER RUN: a fixed name is a race as soon as two refinements are in
    # flight, and the CV family launches five in sequence beside whatever else is running
    drv = os.path.join(SM_ROOT, 'runs', '_refine_driver_%s.py' % run_name)
    with open(drv, 'w', encoding='utf-8') as f:
        f.write(DRIVER % dict(root=SM_ROOT, scripts=SCRIPT_DIR, src=a.src, run_name=run_name, init=('best_pi' if a.init == 'best_pi' else ''),
                              seed=a.seed, lr=a.lr, epochs=a.epochs, patience=a.patience,
                              ny_pde=a.ny_pde, resample=bool(a.lbfgs_resample),
                              adaptive=('None' if a.no_adaptive else repr((a.pool, a.k, a.c))),
                              monitor=a.monitor,
                              num_batches=("cd['num_batches']" if a.num_batches is None
                                           else repr(a.num_batches if a.num_batches == 'auto'
                                                     else int(a.num_batches)))))
    print('\n$ %s %s' % (PY, drv), flush=True)
    r = subprocess.run([PY, drv], cwd=SM_ROOT)
    if r.returncode:
        raise SystemExit(r.returncode)

    print('\nREFINE COMPLETE: runs/v2_nondim/%s' % run_name, flush=True)


if __name__ == '__main__':
    main()
