"""
train_nd.py
===========
Train MIONet v2 - the DIMENSIONLESS single-particle diffusion operator (srcs_EM/mionet_nd.py).

The network learns the parameter-free problem

    dx/dtau = xi^-2 d/dxi ( xi^2 dx/dxi ),     -dx/dxi |_{xi=1} = Phi,     x(xi, 0) = x0(xi)

so ONE trained network serves BOTH electrodes of ANY constant-D_s chemistry at ANY
temperature: chemistry and T enter only through (Phi, tau) = srcs.chemistry.Chemistry.
The electrode sign is inside Phi, which is why there is a single "particle" run.

Training is physics-informed only (PDE / IC / BC residuals + mass conservation + a K-step
recursive rollout); no PyBaMM labels. The existing samplers (srcs.prepData) are reused
unchanged - they already work in [t, xi] in [0,1]^2 - the time column is read as
tau/TAU_MAX and the "current" GRF as Phi. Only the residuals change (TrainerND):

    loss_pde  :  (2 y_xi/xi + y_xixi) - y_t / TAU_MAX        (was  ... - y_t Rs^2/(Ds t_tr))
    loss_bc2  :  y_xi + Phi                                    (was  y_xi + sign i C/j0_aux)
    loss_cons :  <x>(tau) - <x>(0) + 3 Phi tau                 (was  ... + 3 sign i C/j0_aux Fo tau)

Envelope to cover (Mohtat2020, -5..45 C, 0.2..2C): anode |Phi| <= 1.34, cathode <= 0.57,
tau per 60 s window <= 0.142  ->  TAU_MAX = 0.2, Phi ~ N(0, PHI_SF=1.5).

Artifact (runs/v2_nondim/particle[_NNN]/): checkpoints/{best,last}.pt (state_dict payloads),
model.pt (best state_dict ONLY - no pickle), surrogate_manifest.json (architecture + the
trained dimensionless envelope), config.json, metadata.json, history.csv, training.log.

Run button: full training. Set env TRAIN_ND_SMOKE=1 for a 3-epoch pipeline smoke test.
"""
import os
import sys
import json
import re
import shutil
import logging
import datetime
import subprocess

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SM_ROOT    = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, SM_ROOT)
from srcs.prepData import Config, data, xi_warp, xi_nodes, create_loader  # noqa: E402
from srcs.training import Trainer                 # noqa: E402
from srcs.trainingTracker import TrainingTracker  # noqa: E402
from srcs_EM.mionet_nd import MIONetND               # noqa: E402

SMOKE = os.environ.get('TRAIN_ND_SMOKE') == '1'

# ============================ CONFIG ============================
# ---- dimensionless envelope (see docstring) ----
# The defaults below are a base configuration: the reported networks are trained through
# scripts/run_training.py, whose RECIPE and configs/*.json set every value that matters, and
# srcs/hpo_rollout.py imports them too. Comments on individual settings may describe experiments
# with other values.
TAU_MAX   = 0.2          # trained tau window: time column t in [0,1] means tau = t*TAU_MAX.
                         # 0.2 -> 0.1: deployment never exceeds tau = 0.0472 per 20 s step, so half
                         # the old window was never queried while the collocation budget paid for it.
PHI_SF    = 1.5          # Phi ~ N(0, PHI_SF^2) when PHI_DIST='normal'. Phi = GRF * PHI_SF. Widened 1.5 -> 2.5 (2026-09-09): the error grows monotonically
                         # with |Phi| and the cold cases sit at the edge (Mohtat -5C/2C Phi=1.34, Marquis 1.95,
                         # Mohtat -10C/3C ~2.7). N(0,2.5) puts 2.7 at ~1.1 sigma instead of far outside.
# ---- architecture (widths from the HPO'd Negative_MLP_001; Tanh; hard-BC ON) ----
N_SENSORS = 21          # uniform, spacing 0.02. Back to basics: resolve the boundary layer by
                        # RESOLUTION rather than by clustering. delta = sqrt(tau) is 0.040 at
                        # the worst operating point (Marquis anode, 0 C), so 0.02 spacing puts
                        # two nodes inside it without starving the interior the way the
                        # clustered laws did.
HIDDEN    = 128
N_LAYERS  = 5
HARD_BC   = True     # adds -Phi xi^2/2 to the output: the quasi-steady shape under the flux, so the
                     # surface flux is exact by construction and the network learns the rest
ACTIVATION = 'tanh'      # 'tanh' | 'silu' | 'gelu'  (overridden by HPO_CONFIG if set)
TIME_FEATURES = 'linear' # 'linear' = trunk sees (t, xi), as every run so far.
                         # 'linear+sqrt' adds sqrt(t): the short-time surface response is
                         # -2 Phi sqrt(tau/pi), whose derivative is unbounded at tau = 0, and the
                         # Prada anode and Chen cathode query at t = 0.024 and 0.029 of the window.
                         # See MIONetND.TIME_FEATURES.
# ---- physics-informed sampling (as the HPO'd run) ----
N_PDE, NY_PDE = 1500, 10_000    # input functions (profiles x fluxes) and collocation points per profile.
                                 # 1500 x 5000 = 7.5M points/epoch. Raising BOTH to 1500 x 10000 pinned the
                                 # 12 GB card at its ceiling and the epoch time rose 3-10x (shared-memory spill).
N_IC,  NY_IC  = 3000, 100
N_BC,  NY_BC  = 3000, 1000
RESAMPLE_EVERY = 50              # regenerate ALL PI sets (fresh GRF profiles, fluxes, collocation points) every
                                 # N Adam epochs (0 = fixed sets, as v1). A finite fixed set can be overfitted.
NUM_BATCHES   = 38               # int, or 'auto' to size it against the free VRAM (see
                                 # autotune_num_batches: it only RAISES the count when memory
                                 # demands it, never lowers it below NUM_BATCHES_HINT, because the
                                 # count also sets the optimiser steps per epoch).
NUM_BATCHES_HINT = 25            # the floor used by 'auto'
                                 # Keep ~40 profiles per batch. At 25 batches the enlarged N_PDE=1500 gave 60
                                 # profiles/batch and the epoch time went 8.8 s -> 89 s (10x), the signature of
                                 # the driver spilling CUDA allocations into shared host memory on Windows.
SCALE_LENGTH1, SCALE_LENGTH2 = [0.5, 1.0], [0.1, 1.0]    # GRF correlation length of x0, in units of xi.
                         # 0.15 was tried (2026-09-22) on the argument that at 0.5 every drawn profile is a
                         # quarter sine, so the operator only sees the gravest radial mode. It cost ~6x in
                         # validation loss: those modes decay as exp(-(13 pi)^2 tau), i.e. within tau ~ 0.001,
                         # so a roll-out never visits them while the IC term - the largest weight in the
                         # problem - spends its capacity representing them. Reverted.
SCALE_FACTOR1 = 0.2      # GRF part of the initial profile. After PROFILE_AMP_MAX renormalises the
                         # magnitude this sets only the GRF-to-parabola SHAPE ratio, not the size.
PROFILE_LAYER_FRAC = 0.0   # fraction of draws whose GRF is confined to a surface shell by
                           # exp((xi-1)/delta). A GRF of correlation length 0.5..1.0 produces smooth
                           # shapes only, and the parabola is the tau->infinity limit, so nothing in
                           # the sampler looked like a YOUNG profile: flat core, all the drop in the
                           # outer shell. Measured, a best-fit scaled parabola misses the true shape
                           # by 80 % of its depth at tau = 0.002 and 45 % at 0.02. The Prada anode
                           # and Chen cathode - the two worst cells in every run - need 42 and 34
                           # steps of a ~60 step discharge to leave that regime.
PROFILE_LAYER_DELTA = (0.08, 0.35)   # log-uniform shell thickness, delta = sqrt(tau), i.e. the
                           # deployment range tau = 0.006..0.12. The lower bound is set by the BC,
                           # not by tau: the enveloped surface slope is ~PROFILE_AMP_MAX/delta and
                           # the hard-BC ansatz pushes it to -Phi, so below 0.5/(3*2.0) = 0.083 the
                           # IC term would be asking for a slope the BC term fights.
PROFILE_AMP_MAX = None   # peak |x0(xi) - x0(0)|, drawn U(0, this). 0.5 covers 99.84 % of the
                         # 94944 real snapshots (p50 0.031, p90 0.121, p99 0.342, max 0.649) and
                         # leaves training margin above them. 0.2 was tried and reverted: it covers
                         # 95.7 %, which is the deployment range with no headroom at all. The real centre-to-surface spread
                         # over 94944 PyBaMM snapshots is p50 0.031 / p99 0.342 / max 0.649, and 0.5
                         # covers 99.84 % of them with margin left. Uncapped, the parabola inherited
                         # Phi_prev ~ N(0,2) and 29 % of drawn profiles exceeded 1.0 - amplitudes no
                         # stoichiometry profile can have. None = the legacy uncapped sampler.
PHI_DIST      = 'normal'   # 'normal': Phi ~ N(0, PHI_SF^2) as reported. 'uniform': Phi ~ U(-PHI_SPAN*PHI_SF, +)
PHI_SPAN      = 2.0        # half-width of the uniform interval, in units of PHI_SF
PHI_PREV_MATCH = 0.5       # fraction of 'grf+parabola' profiles whose parabola uses the
                           # SAME Phi as the second branch. 0 = parabolic shape kept,
                           # coherence destroyed.
PROFILE_AMP_RANDOM = False   # draw the PDE/IC GRF amplitude from U(0, SCALE_FACTOR1) instead
                             # of fixing it at SCALE_FACTOR1 (the BC sampler always has).
PROFILE_MODE  = 'grf+parabola'   # x0 = GRF + (-Phi_prev xi^2/2): covers deployment profiles (0.03..0.6);
                                 # 'grf' = v1 fixed-amplitude sampler (all ~0.25), which under-covers both ends
                                 # 'shaped' = surface gradient exactly -Phi_prev, parabola -> surface layer
                                 #            (prepData.Config, 2026-10-08); uses the four SHAPED_* below
SHAPED_GRF_FRAC   = 0.2          # 'shaped': fraction of pure-GRF profiles
SHAPED_INDEP_FRAC = 0.3          # 'shaped': fraction with an independent Phi_prev (PHI_PREV_MATCH = matched)
SHAPED_M_RANGE    = (2.0, 20.0)  # 'shaped': exponent m, log-uniform; 2 = parabola, large = surface layer
SHAPED_NOISE      = 0.03         # 'shaped': amplitude of the GRF perturbation on top of the shape
SHAPED_DEPTH_MAX  = 0.65         # 'shaped': depth cap, by redrawing the shape (gradient kept exact)
# ---- radial mesh: ONE switch, then per-decision overrides ----
GEOMETRY      = 'uniform'   # 'uniform' | 'particle', and that is the whole choice. Selecting one
                           # sets the sensors, the PDE and IC collocation points and the
                           # mass-balance nodes together:
                           #   'uniform'  : equally spaced in xi everywhere
                           #   'particle' : uniform to xi = 0.5, then equal VOLUME to the surface
                           # Older, finer-grained laws ('sqrt', pure equal-volume, 'warp', 'block')
                           # are still reachable by setting XI_MAP / XI_COLLOC / XI_QUAD explicitly,
                           # for reproducing an older run. They are not geometry choices.
                           # (original note) the sensors,
                           # the PDE and IC collocation points, the mass-balance quadrature nodes
                           # and the profile space all become the non-uniform, surface-clustered
                           # choice together. A run can no longer end up with warped sensors and
                           # uniform collocation because a field was left behind.
                           #   'uniform'  -> xi_map uniform, xi_colloc uniform, profile_space index
                           #   'particle' -> xi_map surface, xi_colloc GEOMETRY_LAW,
                           #                 xi_quad follows, profile_space physical
GEOMETRY_LAW  = 'hybrid'   # DEPRECATED - kept only so an older config.json still loads. The
                           # geometry switch no longer reads it. Was: the law used for the SENSORS, the PDE/IC collocation points and the
                           # mass-balance nodes under GEOMETRY='particle'. 'particle' = xi = s^(1/3),
                           # density p(xi) ~ xi^2: every sensor, point and node carries the same
                           # VOLUME of active material, which is the measure the lithium inventory,
                           # the SOC and the mass balance all live in. See the note
                           # in srcs/prepData.xi_nodes: 'sqrt' (p(xi)~xi, equal AREA per point) has
                           # the best worst case; 'particle' (p(xi)~xi^2, equal VOLUME) is the
                           # matched rule for a boundary layer but leaves one cell spanning
                           # xi = 0..0.29.
XI_MAP        = 'uniform'      # 'auto' = follow GEOMETRY. # 'uniform': xi_i = i/(N-1) (reported). 'surface': xi_i = 1-(1-s_i)^XI_WARP_P,
                           # clustered where the lithium and the observability are (srcs.prepData.xi_warp)
XI_WARP_P     = 2.0
XI_COLLOC     = 'uniform'      # 'auto' = follow GEOMETRY, which is the rule: one law for the
                           # sensors and for the PDE, IC and mass-balance points alike. It was
                           # briefly decoupled from the sensors; the note below is why that looked
                           # attractive and is worth keeping in view, because on the clustered
                           # geometry the interior still gets little PDE weight.
                           # (original note) DECOUPLED from the sensors on purpose. The sensors are a
                           # REPRESENTATION choice and want resolution where the layer is; the
                           # collocation points are a MEASURE and decide where the PDE is
                           # enforced. Tying them together starved the interior: with the
                           # hybrid law and the volume residual only 3.19 % of the PDE loss
                           # weight falls below xi = 0.5, against 12.5 % on a uniform draw -
                           # and the measured error IS an interior error.
                           # 'auto' = follow GEOMETRY. Radial law for the PDE and IC points:
                           #   'uniform'  : as drawn,        p(xi) ~ 1
                           #   'sqrt'     : xi = s^(1/2),    p(xi) ~ xi    - best worst case
                           #   'particle' : xi = s^(1/3),    p(xi) ~ xi^2  - equal VOLUME per point
                           # (legacy: 'block' = the two-block surface bias, 'warp' = 1-(1-s)^p,
                           #  'auto' = warp on a surface mesh, block otherwise)
                           # tau is ALWAYS uniform - see T_HI.
XI_QUAD       = 'uniform'    # node law for the mass-balance fixed grid: 'uniform' | 'particle' |
                           # 'follow' (use XI_COLLOC). The integrand is 3 xi^2 x, so equal-volume
                           # nodes are the matched rule for a boundary layer - 7x more accurate at
                           # a layer of 0.05 - but ~7x WORSE on a smooth profile, because the law
                           # leaves a single cell spanning xi = 0 .. 0.29. s^(1/2) is the better
                           # compromise if the worst case over both matters more.
T_HI          = 1.0        # time span of the DENSE half of the PDE collocation sampler.
TAU_DENSITY   = (0.0,)     # (q,) = linearly dropping density in tau for the PDE points,
                           # q = density at tau_max / density at 0. None = uniform.
                           # (0.0,) drops to zero at tau_max, (0.25,) keeps a quarter.
                           # It REPLACES the T_HI time box, which is the same idea as a
                           # step function; the two would own the same budget.
TAU_DENSITY_BC = True      # apply TAU_DENSITY to the BC set as well as the PDE set.
TAU_SPLIT     = None       # (frac, hi): DOUBLE DOMAIN IN TIME. A fraction `frac` of every
                           # collocation batch - PDE and BC - is drawn with tau in [0, hi*tau_max]
                           # instead of [0, tau_max]. Orthogonal to the xi treatment, so the xi
                           # sampling is untouched and only the time distribution moves; T_HI's
                           # time block is disabled while it is on, since both would own the same
                           # half of the budget. (0.5, 0.1) = half the points in the first tenth
                           # of the window, which is where deployment lives and where the error
                           # against tau per call rises steeply. None = off.
                           # 1.0 = UNIFORM in tau, which is the right default. The old 0.3 came from
                           # a causality argument - early errors propagate, so weight them - but that
                           # applies to a network that MARCHES through time. This one is a direct map
                           # (x0, Phi, tau) -> x(tau): every tau is an independent query and an error
                           # at tau=0.02 does not propagate to tau=0.08 within one evaluation.
                           # Reducing TAU_MAX to the deployment range already concentrates the window
                           # where it is used, so the extra bias is a distortion on top of that.
                           # 'auto' sizes the dense block to the deployment range for the TAU_MAX in
                           # use; a float fixes it.
TAU_DEPLOY_MAX = 0.0472    # largest tau per 20 s step across the five cells, at 25 C
PROFILE_SPACE = 'physical' # NOT 'auto'. GEOMETRY='uniform' would preset this to 'index', which
                           # draws the GRF at the SENSOR INDICES - so 51 sensors would give a
                           # different ensemble of physical profiles than 24 did, and the mesh
                           # change would stop being a single-variable experiment.
                           # 'auto' = follow GEOMETRY.
                           # 'index' = GRF at the sensor indices (reported); 'physical' = GRF on a fine
                           # uniform xi grid, read at the sensors -> the profile ENSEMBLE is mesh-independent,
                           # which is what makes a mesh change a single-variable experiment
REF_GRID      = 'fv'       # how PyBaMM's radial solution maps onto the sensors: 'legacy' (row k -> k/(N-1),
                           # what every published number used) or 'fv' (true cell centres + surface anchor,
                           # interpolated onto the sensors). 'fv' is mandatory for a non-uniform mesh.
REF_R_PTS     = 32         # PyBaMM radial resolution when REF_GRID='fv'
# ---- optimisation ----
EPOCHS        = 1000
LR            = 4.73e-4
LR_END_FACTOR = 1e-2
# loss weights [pde, ic, bc, cons]. The ND residual is 1/TAU_MAX = 5 (was 1/Fo = 25 at
# t_tr = 50 s), i.e. 5x smaller -> the PDE weight is scaled x5 from the HPO value (0.405).
LOSS_WEIGHTS  = [2.0, 48320.0, 27.3, 10.0]
ROLLOUT_K, W_ROLLOUT, ROLLOUT_N_PTS = 0, 1.0, 300   # K=0 disables the roll-out training term.
                         # Off by the author decision of 2026-09-23: it had never actually been
                         # exercised - run_proof pinned ROLLOUT_K=0 for every run - so it is an
                         # untested term and the back-to-basics run should not carry one.
ROLLOUT_N_PROFILES = 8   # input functions the roll-out chain runs on. It was effectively ONE
                         # (X_ic[:1]), so the only term that can see accumulated drift was trained
                         # on a single operating point per batch. The measured failure is drift:
                         # the Prada anode's CENTRE error grows 0.39 -> 3.16 -> 8.70 -> 10.93 % of
                         # range through a protocol while its surface stays under 3.4 %, and the
                         # lithium-balance projection is a uniform shift that cannot correct a
                         # shape error.
CONS_LEGACY = False      # True restores the two pre-2026-09-23 mass-balance defects exactly:
                         # the initial mean read from the network's own tau=0 output, and the
                         # trapezoid taken in xi rather than in the volume coordinate. Only
                         # for replicating an older run as it actually was.
CONS_USE_IC = False       # also feed the IC batch's profiles to the mass-balance term. It is
                         # forward-only, so tripling its input functions costs a few per cent.
# mass-balance term: the fixed radial grid (law = XI_QUAD) and the number of tau stations, swept
# deterministically as tau_i = TAU_MAX * i/N. N = 4 sampled the window too coarsely to constrain
# the shape of the drift; 10 costs 10 forward passes per call on a 41-node grid, which is cheap
# next to the PDE term's second-order autograd.
CONS_N_KSI, CONS_N_T = 41, 4
PATIENCE      = 100
LOG_EVERY     = 10
PDE_FORM      = 'conservative'         # 'conservative' | 'volume' | 'strong'. See _pde_residual.
                         # NO weighting: res = r, so the loss is the plain L2 norm of the PDE
                         # residual and the interior carries 50 % of it on a uniform draw
                         # against 3.1 % for 'volume' and 0.03 % for 'conservative'.
                         # The 1/xi term is safe here: the sampler clamps xi >= 0.01, and on an
                         # untrained net only 1.9 % of the residual MSE comes from xi < 0.02.
                         # Earlier note on the other two forms:
                         # 'volume' because the conservative residual is xi^2*r, so its square puts
                         # xi^4 in the loss and leaves only 0.7-3 % of the PDE weight below
                         # xi = 0.5 - while the measured error is an INTERIOR error, 2-4x larger at
                         # xi < 0.2 than anywhere else and 89-100 % network rather than mesh.
                         # Original note on the two older forms:   # 'conservative': xi^2*y_t/tau_max - (xi^2*y_xi)_xi  (PDE x xi^2: no 1/xi
                         # singularity, volume-weighted). 'strong': 2 y_xi/xi + y_xixi - y_t/tau_max (v1 form; its
                         # MSE is dominated by the near-centre band - 175x the bulk on the best net - so L-BFGS
                         # minimised it while the held-out error rose).
INIT_FROM     = None     # warm start: a run folder or a .pt of a state_dict. The architecture
                         # must match; nothing else about the run changes.
RESUME        = False    # continue the interrupted run RUN_BASE/RUN_NAME from its latest checkpoint
                         # (scripts/run_training.py --resume). The same settings must be given as for
                         # the original launch; see resume_state for what makes the continuation exact.
ONLY_LBFGS    = False    # skip Adam entirely and go straight to L-BFGS. Only sensible with
                         # INIT_FROM - L-BFGS from a random initialisation is a poor idea.
ADAPTIVE_SAMPLE = None   # (pool_mult, k, c): RESIDUAL-WEIGHTED collocation. Draw pool_mult times
                         # the point budget, score every candidate with the warm-started network,
                         # and keep Ny_pde of them with probability proportional to
                         #     p(x) = |r(x)|^k / E[|r|^k] + c
                         # (Wu et al.'s RAD). c > 0 keeps a uniform floor so the sampler cannot
                         # abandon the regions it currently satisfies - without it the residual
                         # migrates to wherever the points left. Needs INIT_FROM to be meaningful:
                         # a random network's residual map is noise.
LR_DECAY_EPOCHS = None  # int: the linear decay to LR_END_FACTOR spans this many epochs (instead of EPOCHS);
                        # afterwards the lr stays frozen at LR_END_FACTOR * LR. Needs LR_DECAY.
ES_FROM_EPOCH = 0       # early stopping (PATIENCE epochs without improvement) is only checked from this
                        # epoch on; 0 = always, as in every earlier run
LR_DECAY      = False    # True: decay the Adam lr linearly to LR_END_FACTOR over EPOCHS (the planned
                         # epochs) also without an in-loop L-BFGS switch. False keeps the shipped m21d
                         # behaviour: the scheduler only steps when LBFGS_PHASE is on, so lr is constant.
LBFGS_PHASE   = False    # OFF. Measured on hyb_unicol_vol_roll: 5 L-BFGS epochs, ~50 min, each
                         # slower than the last (320 -> 539 -> 693 -> 794 s), and the final
                         # artifact still came from the ADAM checkpoint at epoch 124 - it never
                         # beat the monitored metric once. It only moved pi_val_loss. Turn it
                         # back on for a run where the Adam phase has visibly converged.
                         # (original note) at the first early stop switch Adam -> L-BFGS.
LBFGS_PATIENCE = 5       # epochs of no improvement AFTER the switch before the run really stops.
                         # It used to be hardcoded at 20. L-BFGS here is a refinement, not a search:
                         # on particle_001 it took the held-out error from 1.27e-4 to 1.85e-5 within
                         # SIX epochs, and on the runs where it hurt it did so immediately, driving
                         # its own objective down while the held-out error rose. Either way the
                         # verdict is in well inside 5 epochs, and each one costs ~14 min full-batch.
LR_LBFGS      = 1e-3     # initial step of the strong-Wolfe L-BFGS (base default 1e-2 diverged on pure-PI runs)
LBFGS_FULL_BATCH = True  # L-BFGS on the FULL data set (one deterministic batch per epoch). With mini-batches the
                         # curvature pairs of one batch are applied to the next: pure-PI runs drifted 4e-5 -> 8e-4
                         # within 10 epochs at lr 1e-3 as well as 1e-2. PDE points are sub-sampled per profile
                         # to LBFGS_NY_PDE (memory: second-order autograd over N_PDE x LBFGS_NY_PDE points).
LBFGS_RESAMPLE = True    # re-draw the full-batch point subset EVERY L-BFGS epoch and reset the L-BFGS history:
                         # on a FIXED subset L-BFGS drove its objective down 4x while the held-out error rose
                         # (particle_pi_conservative: it overfits the collocation sample, Adam saw 10k pts/profile)
LBFGS_NY_PDE, LBFGS_NY_BC, LBFGS_NY_IC = 1000, 200, 100  # points per profile in the full batch (pde: 2nd-order
                         # autograd; the FULL bc/ic sets (1M/100k pts) made one L-BFGS epoch > 50 min at 12 GB)
                         # Slow (~14 min/epoch on the 4070 at full size).
# ---- validation / model selection: a held-out PHYSICS-INFORMED set, never trained on ----
TRACK_PI_VAL = True      # compute the weighted PI loss on a held-out collocation set every epoch
MONITOR = 'pi_val_loss'  # what early stopping and best.pt follow: the held-out physics loss
MONITOR2 = None          # optionally a SECOND metric to keep a best checkpoint for, saved beside
                         # best.pt and exported to <run>/best_pi/ (None: none)
# ~10 % of the training set, drawn from the SAME distribution but never trained on, and scored
# with the SAME loss weights - so pi_val_loss is directly comparable with the training PI loss and
# the gap between them is the physics-side generalisation gap.
VAL_PI_SIZES = dict(n_pde=150, ny_pde=1000, n_ic=300, ny_ic=100, n_bc=300, ny_bc=100)
CACHE_DIR = os.path.join(SM_ROOT, 'runs', 'hpo_cache')
# ---- bookkeeping ----
RUN_BASE  = os.path.join(SM_ROOT, 'runs', 'v2_nondim')
RUN_NAME  = 'particle'
HPO_CONFIG = None        # path to runs/hpo_nd*/model_config.json (from srcs/hpo_rollout.py) ->
                         # overrides HIDDEN / N_LAYERS / ACTIVATION / LR / LOSS_WEIGHTS below
SAMPLER_PARAM_SET = 'Mohtat2020'   # only used by the sampler for C_Ah; the net never sees it
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED      = 42
# =================================================================
if SMOKE:                       # tiny end-to-end pipeline check
    N_PDE, NY_PDE, N_IC, NY_IC, N_BC, NY_BC = 40, 1500, 60, 50, 60, 200
    NUM_BATCHES, EPOCHS, PATIENCE, LOG_EVERY = 4, 3, 10, 1
    if os.environ.get('TRAIN_ND_SMOKE_LBFGS') == '1':       # force the L-BFGS switch on epoch 1
        EPOCHS, PATIENCE = 4, 1
    RUN_NAME = 'particle_smoke'
if HPO_CONFIG:                  # apply an HPO result (architecture + lr + PI loss weights)
    _h = json.load(open(HPO_CONFIG))
    HIDDEN, N_LAYERS, ACTIVATION = int(_h['hidden_dim']), int(_h['number_of_hidden_layers']), _h['activation']
    LR, LOSS_WEIGHTS = float(_h['lr']), [float(w) for w in _h['loss_weights']]

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def resolved_xi_map(xi_map=None):
    """XI_MAP with 'auto' resolved against GEOMETRY -- the same rule srcs.prepData applies."""
    m = xi_map or XI_MAP
    return ('uniform' if GEOMETRY == 'uniform' else GEOMETRY_LAW) if m == 'auto' else m


def sensor_grid(n_sensors=None, xi_map=None, p=None):
    """The xi positions of the branch-1 inputs -- the single definition that the sampler, the trainer
    and the saved manifest all have to agree on."""
    m, ns = resolved_xi_map(xi_map), int(n_sensors or N_SENSORS)
    if m == 'surface':
        return xi_warp(np.linspace(0, 1, ns), p if p is not None else XI_WARP_P)
    return xi_nodes(ns, m)


class TrainerND(Trainer):
    """Trainer with the DIMENSIONLESS residuals. Everything else (IC, BC1, rollout,
    optimiser, scheduler, tracker plumbing) is inherited unchanged."""

    def __init__(self, model, cfg, tracker, tau_max, val_pi, lr_lbfgs=1e-2,
                 lbfgs_full_batch=False, lbfgs_ny=(1000, 1000, 100), pde_form='conservative', lbfgs_resample=False):
        super().__init__(model, cfg, tracker=tracker)      # loads PyBaMM params; unused here
        self.tau_max = float(tau_max)
        self.val_pi = tuple(x.to(self.device) for x in val_pi)   # held-out (pde, ic, bc)
        self.lr_lbfgs = float(lr_lbfgs)                                     # L-BFGS phase initial step
        if pde_form not in ('conservative', 'volume', 'strong'):
            raise ValueError("pde_form must be 'conservative', 'volume' or 'strong'")
        self.pde_form = pde_form
        self.lbfgs_full_batch, self.lbfgs_ny = bool(lbfgs_full_batch), tuple(int(n) for n in lbfgs_ny)
        self.lbfgs_resample, self._lbfgs_src = bool(lbfgs_resample), None

    def _full_batch_loaders(self, loader_pde, loader_ic, loader_bc, loader_val, seed=0):
        """Concatenate every mini-batch into ONE batch per loader for the L-BFGS phase (deterministic
        objective within an epoch). Points are sub-sampled per profile to `lbfgs_ny` (seeded: with
        lbfgs_resample a fresh subset is drawn each epoch)."""
        if self._lbfgs_src is None:
            self._lbfgs_src = (torch.cat([b[0] for b in loader_pde]), torch.cat([b[0] for b in loader_bc]),
                               torch.cat([b[0] for b in loader_ic]),
                               [loader_val[0]])
        Xp_all, Xb_all, Xi_all, Xv = self._lbfgs_src
        g = torch.Generator().manual_seed(int(seed))
        def sub(X, n):                                             # fixed random subset of the points per profile
            return X if X.shape[1] <= n else X[:, torch.randperm(X.shape[1], generator=g)[:n], :]
        Xp, Xb, Xi = sub(Xp_all, self.lbfgs_ny[0]), sub(Xb_all, self.lbfgs_ny[1]), sub(Xi_all, self.lbfgs_ny[2])
        logger.info("L-BFGS full batch: pde %s | ic %s | bc %s", tuple(Xp.shape), tuple(Xi.shape), tuple(Xb.shape))
        return [(Xp,)], [(Xi,)], [(Xb,)], Xv

    def _pde_residual(self, y_t, y_k, y_kk, ksi):
        """The PDE residual in the form `pde_form` selects. Shared by loss_pde and the roll-out
        term, which used to hardcode the strong form and so disagreed with the main loss whenever
        pde_form was anything else."""
        if self.pde_form == 'conservative':
            return (2 * ksi * y_k + ksi ** 2 * y_kk) - ksi ** 2 * y_t / self.tau_max
        if self.pde_form == 'volume':
            return (2 * y_k + ksi * y_kk) - ksi * y_t / self.tau_max
        return (2 * y_k / ksi + y_kk) - y_t / self.tau_max

    def loss_pde(self, X):
        t = X[:, :, :1].requires_grad_()
        ksi = X[:, :, 1:2].requires_grad_()
        cs_0 = X[:, :1, 2:2 + self.b1]
        phi = X[:, :1, 2 + self.b1:2 + self.b1 + self.b2]
        y = self.network(t, ksi, cs_0, phi)
        y_t = torch.autograd.grad(y, t, torch.ones_like(y), create_graph=True)[0]
        y_k = torch.autograd.grad(y, ksi, torch.ones_like(y), create_graph=True)[0]
        y_kk = torch.autograd.grad(y_k, ksi, torch.ones_like(y_k), create_graph=True)[0]
        return torch.mean(self._pde_residual(y_t, y_k, y_kk, ksi) ** 2)

    def pde_residual_abs(self, X):
        """|PDE residual| at every collocation point: (B, Ny). Same residual loss_pde squares and
        averages - kept as one expression so the two can never drift apart."""
        t = X[:, :, :1].requires_grad_()
        ksi = X[:, :, 1:2].requires_grad_()
        cs_0 = X[:, :1, 2:2 + self.b1]
        phi = X[:, :1, 2 + self.b1:2 + self.b1 + self.b2]
        y = self.network(t, ksi, cs_0, phi)
        # order matters: y_kk needs y_k's graph, and y_t still needs y's, so the first two calls
        # keep the graph alive and only the last one frees it. Nothing is backpropagated to the
        # parameters here - this is a measurement, not a step.
        y_k = torch.autograd.grad(y, ksi, torch.ones_like(y), create_graph=True)[0]
        y_kk = torch.autograd.grad(y_k, ksi, torch.ones_like(y_k), retain_graph=True)[0]
        y_t = torch.autograd.grad(y, t, torch.ones_like(y))[0]
        return self._pde_residual(y_t, y_k, y_kk, ksi).abs().detach().squeeze(-1)

    def loss_bc2(self, X):
        t = X[:, :, :1]
        ksi = torch.ones_like(t).requires_grad_()
        cs_0 = X[:, :1, 2:2 + self.b1]
        phi = X[:, :1, 2 + self.b1:2 + self.b1 + self.b2]
        y = self.network(t, ksi, cs_0, phi)
        y_k = torch.autograd.grad(y, ksi, torch.ones_like(y), create_graph=True)[0]
        return torch.mean((y_k + phi) ** 2)                 # -dx/dxi|_1 = Phi

    def _cons_batch(self, X_pde, X_ic):
        """Input functions for the mass balance.

        The term only reads a profile and its flux, and costs one forward pass per tau station -
        no autograd at all - so restricting it to the PDE batch was leaving free constraint on the
        table. Adding the IC batch roughly triples the number of (x0, Phi) pairs it sees per step
        at a few per cent of the epoch cost.
        """
        c = 2 + self.b1 + self.b2
        parts = [X_pde[:, :1, :c]]
        if getattr(self, 'cons_use_ic', False) and X_ic is not None and X_ic.shape[2] >= c:
            parts.append(X_ic[:, :1, :c])
        return parts[0] if len(parts) == 1 else torch.cat(parts, 0)

    def loss_cons(self, X, n_ksi=None, n_t=None):
        """<x>(tau) - <x>(0) = -3 Phi tau   (divergence theorem + BC), tau = t * TAU_MAX.

        Two things this does NOT do.

        It does not evaluate the network at tau = 0 to get <x>(0). That value is known exactly -
        it is the volume mean of the input profile, which the network is separately required to
        reproduce by the IC loss. Taking it from the network instead made the constraint relative
        to the network's own imperfect tau=0 output, so a model could conserve mass perfectly
        about the WRONG starting value and still score zero here. Using the exact initial mean
        ties the balance to the true inventory and removes one forward pass per call.

        It does not integrate on a uniform grid unless asked to. The integrand is 3 xi^2 x, so
        the quadrature error is dominated by the outer shells, exactly where the boundary layer
        sits: on 41 uniform nodes a layer of thickness 0.05 integrates to 3.4e-3 against a
        residual of ~0.15, i.e. the term partly fits its own integration error. The 'particle'
        node law puts equal VOLUME between nodes, which is the measure this integral lives in.
        """
        B = X.shape[0]
        n_ksi = int(getattr(self, 'cons_n_ksi', 41) if n_ksi is None else n_ksi)
        n_t = int(getattr(self, 'cons_n_t', 4) if n_t is None else n_t)
        cs_0 = X[:, :1, 2:2 + self.b1]
        phi = X[:, :1, 2 + self.b1:2 + self.b1 + self.b2]
        phi_s = phi.reshape(B, -1)[:, :1]
        nodes = xi_nodes(n_ksi, getattr(self, 'xi_quad', 'uniform'),
                         p=getattr(self, 'xi_warp_p', 2.0))
        nodes[0] = max(nodes[0], 1e-6)                      # keep xi=0 off the exact singularity
        ksi = torch.as_tensor(nodes, device=X.device, dtype=X.dtype).view(1, -1, 1).expand(B, -1, -1)
        # Integrate in the VOLUME coordinate u = xi^3, not in xi. The two are the same integral -
        #     <x> = 3 int x xi^2 dxi = int x du,     u = xi^3, du = 3 xi^2 dxi
        # - but not the same quadrature. In xi a trapezoid has to resolve the xi^2 weight as well
        # as the profile; in u the integrand is just x, unweighted, and on equal-volume nodes the
        # u_i are uniform. Measured on 604 real PyBaMM profiles at 41 nodes, the worst error of
        # <x> falls 1.14e-2 -> 1.80e-4 for the 'particle' law, 2.1e-3 -> 8.6e-5 for 'sqrt' and
        # 3.0e-4 -> 1.1e-4 for 'uniform'. The old pairing mattered: 1.14e-2 is the size of the
        # constraint itself (3 Phi tau ~ 1.2e-2 at a 20 s step), so the term was largely fitting
        # its own integration error - and it penalised the equal-volume law hardest, which is the
        # one whose whole purpose is to make this integral easy.
        # CONS_LEGACY restores the two pre-2026-09-23 behaviours exactly, so a run can be
        # replicated as it was. Both are defects, kept reachable only for that purpose:
        #   - avg0 from the NETWORK's own tau=0 output, so the balance is measured against the
        #     model's own imperfect starting value and can score zero about the WRONG mean;
        #   - the trapezoid taken in xi against the 3 xi^2 weight rather than in the volume
        #     coordinate, which was 3x worse on real profiles and 63x worse on equal-volume nodes.
        legacy = bool(getattr(self, 'cons_legacy', False))
        u = ksi if legacy else ksi ** 3
        w = 3.0 * ksi ** 2 if legacy else 1.0
        if legacy:
            avg0 = torch.trapezoid(w * self.network(torch.zeros_like(ksi), ksi, cs_0, phi), u, dim=1)
        else:
            avg0 = torch.trapezoid(self._linear_interp(cs_0, ksi), u, dim=1)     # exact initial mean
        # tau_i = tau_max * i/N: a deterministic sweep of the whole trained window. It used to be
        # scaled by the largest t in the sampled batch, which made the constraint depend on the
        # draw and, with a front-loaded time sampler, never reached the end of the window at all.
        res = []
        for f in range(1, n_t + 1):
            t = f / n_t
            avg = torch.trapezoid(w * self.network(torch.full_like(ksi, float(t)), ksi, cs_0, phi),
                                  u, dim=1)
            res.append(avg - avg0 + 3.0 * phi_s * self.tau_max * t)
        return torch.mean(torch.stack(res) ** 2)

    def loss_pi_rollout(self, X_pde, X_ic, X_bc, K=3, detach_cs=False, create_graph=True):
        """K-step recursive physics-informed loss with the DIMENSIONLESS residuals.

        MUST be overridden: the base Trainer re-implements the residuals INLINE with the
        v1 dimensional constants - `y_t*Rs^2/(Ds*t_tr)` and `sign*I*C/j0_aux`, evaluated
        for ONE electrode at 298 K - which would silently pull the dimensionless net toward
        the wrong PDE (~250x off in the time scaling, ~10x in the flux, anode-specific).
        Same structure as the base method; only the two residuals differ:
            pde:  (2 y_xi/xi + y_xixi) - y_t / tau_max        bc2:  dy/dxi|1 + Phi
        """
        device, dtype = self.device, X_ic.dtype
        # B input functions, not one. The chain used to run on X_ic[:1] - a SINGLE profile per
        # batch - so the only term that sees accumulated drift was trained on one operating point
        # at a time. Measured drift is the reason the term is on at all: on the Prada anode the
        # centre error grows 0.39 -> 10.93 % of range through a protocol.
        B = max(1, min(int(getattr(self, 'rollout_n_profiles', 1)),
                       X_pde.shape[0], X_ic.shape[0], X_bc.shape[0]))
        X_pde, X_ic, X_bc = X_pde[:B], X_ic[:B], X_bc[:B]
        cs_0 = X_ic[:, :1, 2:2 + self.b1]                                      # (B, 1, b1)
        losses = []
        for k in range(K):
            row = k % X_bc.shape[1]
            phi_k = X_bc[:, row:row + 1, 2 + self.b1:2 + self.b1 + self.b2]   # (B, 1, 1) = Phi
            # PDE residual (dimensionless)
            t_p = X_pde[:, :, :1].requires_grad_(); ksi_p = X_pde[:, :, 1:2].requires_grad_()
            y_p = self.network(t_p, ksi_p, cs_0, phi_k)
            y_t = torch.autograd.grad(y_p, t_p, torch.ones_like(y_p), create_graph=create_graph)[0]
            y_k = torch.autograd.grad(y_p, ksi_p, torch.ones_like(y_p), create_graph=True)[0]
            y_kk = torch.autograd.grad(y_k, ksi_p, torch.ones_like(y_k), create_graph=create_graph)[0]
            l_pde = torch.mean(self._pde_residual(y_t, y_k, y_kk, ksi_p) ** 2)
            # IC: network at t=0 reproduces cs_0
            ksi_ic = X_ic[:, :, 1:2]
            y_ic = self.network(torch.zeros_like(ksi_ic), ksi_ic, cs_0, phi_k)
            l_ic = torch.mean((y_ic - self._linear_interp(cs_0, ksi_ic)) ** 2)
            # BC1: centre symmetry
            t_bc = X_bc[:, :, :1]
            ksi_b1 = torch.zeros_like(t_bc).requires_grad_()
            y_b1 = self.network(t_bc, ksi_b1, cs_0, phi_k)
            l_bc1 = torch.mean(torch.autograd.grad(y_b1, ksi_b1, torch.ones_like(y_b1),
                                                   create_graph=create_graph)[0] ** 2)
            # BC2: -dx/dxi|1 = Phi
            ksi_b2 = torch.ones_like(t_bc).requires_grad_()
            y_b2 = self.network(t_bc, ksi_b2, cs_0, phi_k)
            dy_b2 = torch.autograd.grad(y_b2, ksi_b2, torch.ones_like(y_b2), create_graph=create_graph)[0]
            l_bc2 = torch.mean((dy_b2 + phi_k) ** 2)
            losses.append(self.W_pde * l_pde + self.W_ic * l_ic + self.W_bc * (l_bc1 + l_bc2))
            # advance one window (t = 1 <=> tau = tau_max) and chain
            xi_s = torch.as_tensor(self.sensors1, device=device, dtype=dtype).reshape(
                1, self.b1, 1).expand(B, -1, -1)
            t_end = torch.ones(B, self.b1, 1, device=device, dtype=dtype)
            cs_0 = self.network(t_end, xi_s, cs_0, phi_k).permute(0, 2, 1)     # (B, 1, b1)
            if detach_cs:
                cs_0 = cs_0.detach()
        return sum(losses) / K

    def pi_val_loss(self):
        """The training loss, same weights, on the HELD-OUT collocation set.

        Cheap relative to a training step: the PDE residual still needs derivatives with respect to
        the INPUTS, so grad has to be enabled, but nothing is backpropagated to the parameters and
        no optimiser state is touched. It tells whether the network still satisfies the physics on
        collocation points it has not seen: a gap opening between it and the training loss is the
        signature of overfitting the finite collocation sample."""
        if self.val_pi is None:
            return 0.0
        Xp, Xi, Xb = self.val_pi
        with torch.enable_grad():
            l = (self.W_pde * self.loss_pde(Xp) + self.W_ic * self.loss_ic(Xi)
                 + self.W_bc * (self.loss_bc1(Xb) + self.loss_bc2(Xb)))
            if self.W_cons > 0:
                l = l + self.W_cons * self.loss_cons(Xp)
        v = float(l.detach())
        self.network.zero_grad(set_to_none=True)
        return v

    def cons_val_loss(self):
        """UNWEIGHTED lithium balance on the held-out collocation profiles, as an RMS.

        `cons_loss` in history.csv is the training value and carries loss_weights[3] in front of
        it, so it cannot be read as a physical quantity and cannot be compared between runs with
        different weights. This is the same residual - <x>(tau) - <x>(0) + 3 Phi tau, in
        stoichiometry units - on profiles the network was not trained on, with no weight. Against
        it, the size of the constraint itself at a 20 s step is ~1.2e-2.

        No input derivatives are involved, unlike the PDE residual, so it runs under no_grad.
        """
        if self.val_pi is None:
            return 0.0
        with torch.no_grad():
            return float(self.loss_cons(self.val_pi[0]).detach().sqrt())


    def train(self, loader_pde, loader_ic, loader_bc, loader_val, only_LBFGS=False, lr_lbfgs=1e-2,
              resample_fn=None, resample_every=0, start_epoch=0):
        """The training loop. The held-out validation metrics are computed once per epoch on the
        END-OF-EPOCH state, so the tracker's pi_val_loss / best checkpoint refer to the weights
        actually saved. start_epoch > 0 continues a resumed run (see resume_state)."""
        import math
        from tqdm import tqdm
        if only_LBFGS:
            self.replace_optimizer(lr_lbfgs)
            # LBFGS_FULL_BATCH was only ever honoured on the Adam -> L-BFGS SWITCH path, so a run
            # that started in L-BFGS silently got MINI-BATCH L-BFGS: the curvature history is built
            # on one batch and then applied to the next, whose objective is different, so the line
            # search fails and the run freezes. particle_proof_v1_mid froze completely after epoch
            # 0 this way, while a single full-batch L-BFGS call from the same weights moves the
            # objective 36 %.
            if self.lbfgs_full_batch:
                loader_pde, loader_ic, loader_bc, loader_val = self._full_batch_loaders(
                    loader_pde, loader_ic, loader_bc, loader_val)
            logger.info("Training with only LBFGS optimizer. | full_batch=%s resample=%s",
                        self.lbfgs_full_batch, self.lbfgs_resample)

        def closure():
            self.optimizer.zero_grad()
            loss_pde = self.loss_pde(X_pde) * self.W_pde
            loss_ic = self.loss_ic(X_ic) * self.W_ic
            loss_bc1 = self.loss_bc1(X_bc) * self.W_bc
            loss_bc2 = self.loss_bc2(X_bc) * self.W_bc
            loss = loss_pde + loss_ic + loss_bc1 + loss_bc2
            if self.W_cons > 0:
                loss_cons_term = self.loss_cons(self._cons_batch(X_pde, X_ic)) * self.W_cons
                loss = loss + loss_cons_term
                cons_loss.append(loss_cons_term.item())
            if self.rollout_K > 0:
                detach = (epoch / max(1, self.num_epochs)) < 0.2
                n_pts = self.rollout_n_pts
                nb = max(2, int(getattr(self, 'rollout_n_profiles', 2)))
                loss_rollout = self.loss_pi_rollout(X_pde[:nb, :n_pts], X_ic[:nb, :n_pts],
                                                    X_bc[:nb, :n_pts],
                                                    K=self.rollout_K, detach_cs=detach)
                loss = loss + self.w_rollout * loss_rollout
            loss.backward()
            pde_loss.append(loss_pde.item()); ic_loss.append(loss_ic.item())
            bc_loss.append(loss_bc1.item() + loss_bc2.item())
            return loss

        self.network.train()
        for epoch in tqdm(range(start_epoch, self.num_epochs), desc='Training : ', initial=start_epoch,
                          total=self.num_epochs):
            if (resample_fn is not None and resample_every > 0 and epoch > 0 and epoch % resample_every == 0
                    and not isinstance(self.optimizer, torch.optim.LBFGS)):
                loader_pde, loader_ic, loader_bc = resample_fn()          # fresh input functions + collocation points
                logger.info("epoch %d: PI sets resampled (profiles, fluxes, collocation points)", epoch)
            if self.lbfgs_resample and self._lbfgs_src is not None and isinstance(self.optimizer, torch.optim.LBFGS):
                loader_pde, loader_ic, loader_bc, loader_val = self._full_batch_loaders(loader_pde, loader_ic, loader_bc, loader_val, seed=epoch)
                self.optimizer = torch.optim.LBFGS(self.network.parameters(), lr=self.lr_lbfgs, max_iter=100,
                                                   tolerance_grad=1e-6, tolerance_change=1e-9, line_search_fn='strong_wolfe')
            epoch_loss, n = 0.0, 0
            pde_loss, bc_loss, ic_loss, cons_loss = [], [], [], []
            for batch_pde, batch_ic, batch_bc, X_val in zip(loader_pde, loader_ic, loader_bc, loader_val):
                X_pde = batch_pde[0].to(self.device); X_ic = batch_ic[0].to(self.device)
                X_bc = batch_bc[0].to(self.device); X_val = X_val.to(self.device)
                loss = self.optimizer.step(closure)
                epoch_loss += loss.item(); n += 1
            epoch_loss /= max(1, n)
            # ONCE, on the end-of-epoch state
            pi_val = self.pi_val_loss() if getattr(self, 'track_pi_val', False) else None
            cons_val = self.cons_val_loss()      # always tracked: it is the term that matters
            pde_loss, ic_loss, bc_loss = np.mean(pde_loss), np.mean(ic_loss), np.mean(bc_loss)
            # the lithium-balance term (loss_weights[3]) was computed every batch but never
            # recorded, so the one loss component the paper singles out as mattering most was
            # missing from every history.csv. It is the "Mass Balance" curve in fig_training.
            cons_loss = np.mean(cons_loss) if cons_loss else 0.0

            if epoch > 0.8 * self.num_epochs and self.change_opt:
                self.replace_optimizer(self.lr_lbfgs); logger.info("Optimizer change at iteration %d (L-BFGS lr %.1e).", epoch, self.lr_lbfgs)
                if self.lbfgs_full_batch:
                    loader_pde, loader_ic, loader_bc, loader_val = self._full_batch_loaders(loader_pde, loader_ic, loader_bc, loader_val)
            if math.isnan(loss.item()):
                logger.warning("NaN detected at iteration %d, stopping loop.", epoch); break
            if self.tracker is not None:
                extra = {} if pi_val is None else dict(pi_val_loss=pi_val)
                extra['cons_val'] = cons_val
                self.tracker.update(epoch, self.network, self.optimizer, train_loss=epoch_loss,
                                    pde_loss=pde_loss, ic_loss=ic_loss, bc_loss=bc_loss,
                                    cons_loss=cons_loss, **extra)
                if self.tracker.early_stop_check(epoch=epoch):
                    if self.change_opt:
                        self.replace_optimizer(self.lr_lbfgs); logger.info("Optimizer change at iteration %d (L-BFGS lr %.1e).", epoch, self.lr_lbfgs)
                        if self.lbfgs_full_batch:
                            loader_pde, loader_ic, loader_bc, loader_val = self._full_batch_loaders(loader_pde, loader_ic, loader_bc, loader_val)
                        self.tracker.best_epoch = epoch
                        self.tracker.patience = int(getattr(self, 'lbfgs_patience', 20))
                    else:
                        break
            else:
                logger.info("Epoch %d | Train Loss: %.4f | PI val loss: %s", epoch, epoch_loss, pi_val)
            if self.scheduler is not None and (self.change_opt or getattr(self, 'lr_decay', False)):
                self.scheduler.step()
        if self.tracker is not None:
            self.tracker.close()


def autotune_num_batches(trainer, X_pde, X_ic, X_bc, n_batches_hint=25, safety=0.70,
                         probes=(2, 6), device='cuda', never_below_hint=True):
    """Fewest batches (largest batch) whose MEASURED peak allocation fits the free VRAM.

    This is measured rather than guessed because the failure it prevents is silent. On Windows an
    over-large batch does not raise OutOfMemory: the WDDM driver spills CUDA allocations into
    shared host memory and the epoch time rises by an order of magnitude with no error. Catching
    exceptions would therefore find nothing - the peak allocation has to be compared against the
    free memory directly.

    Peak cost of one training step is close to affine in the number of profiles per batch,
    peak = a + b*P, so two small probes determine it and the largest affordable P follows. Only
    `free` memory is used as the budget, so a card shared with another job is handled correctly.
    """
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        return n_batches_hint
    n_pde = X_pde.shape[0]
    share_ic, share_bc = X_ic.shape[0] / n_pde, X_bc.shape[0] / n_pde

    def peak_for(P):
        """Incremental peak allocation of one closure at P profiles per batch."""
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        xp = X_pde[:P].to(device)
        xi = X_ic[:max(1, int(round(P * share_ic)))].to(device)
        xb = X_bc[:max(1, int(round(P * share_bc)))].to(device)
        loss = (trainer.loss_pde(xp) * trainer.W_pde + trainer.loss_ic(xi) * trainer.W_ic
                + trainer.W_bc * (trainer.loss_bc1(xb) + trainer.loss_bc2(xb)))
        if trainer.W_cons > 0:
            loss = loss + trainer.loss_cons(xp) * trainer.W_cons
        loss.backward()
        pk = torch.cuda.max_memory_allocated() - base
        trainer.network.zero_grad(set_to_none=True)
        del xp, xi, xb, loss
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        return pk

    free_before, total = torch.cuda.mem_get_info()
    try:
        p1, p2 = probes
        m1, m2 = peak_for(p1), peak_for(p2)
    except Exception as exc:                      # a probe failing is not a reason to stop training
        logger.warning("batch autotune failed (%s) - falling back to %d batches", exc, n_batches_hint)
        return n_batches_hint
    b = max((m2 - m1) / float(p2 - p1), 1.0)       # bytes per extra profile
    a = max(m1 - b * p1, 0.0)
    budget = safety * free_before
    p_max = int(max(1.0, (budget - a) / b))
    nb = int(max(1, min(n_pde, -(-n_pde // p_max))))          # ceil division
    # The batch count is not only a memory knob: it sets the number of optimiser steps per epoch,
    # so lowering it below the tuned value would change the training dynamics as well as the
    # footprint. By default this acts as a SAFETY CAP - it raises the batch count when memory
    # demands it and otherwise leaves the tuned value alone, so runs stay comparable.
    nb_fit = nb
    if never_below_hint:
        nb = max(nb, int(n_batches_hint))
    logger.info("batch autotune: free %.2f GB of %.2f GB | fixed %.0f MB + %.1f MB per profile"
                "  -> %d profiles/batch fits at %.0f%% budget -> %d batches needed; using %d%s",
                free_before / 2**30, total / 2**30, a / 2**20, b / 2**20,
                min(p_max, n_pde), 100 * safety, nb_fit, nb,
                ' (floor: the configured %d, to keep the optimiser dynamics comparable)' % n_batches_hint
                if nb != nb_fit else '')
    return nb


def _git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=SM_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


# Resolved at IMPORT, which is when Python fixes the code that will actually run. Stamping it when
# the artifact is written instead recorded commits the run never executed (see the 2026-09-23
# worklog entry): a run takes tens of minutes and the tree moves underneath it.
LAUNCH_HASH = _git_hash()


def load_weights(model, src):
    """Warm start from a run folder or a state_dict file."""
    path = src if os.path.isfile(src) else os.path.join(src, 'model.pt')
    if not os.path.exists(path):
        path = os.path.join(src, 'checkpoints', 'best.pt')
    sd = torch.load(path, map_location='cpu', weights_only=True)
    sd = sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd
    model.load_state_dict(sd)
    logger.info("INIT_FROM %s : %d tensors loaded", path, len(sd))


def _chunks(n, size):
    for i in range(0, n, size):
        yield i, min(i + size, n)


def adaptive_resample(trainer, Data, X_pde, pool_mult=4, k=1.0, c=1.0, device='cuda', chunk=64):
    """Residual-weighted collocation (RAD), per input function.

    A pool of pool_mult x the budget is drawn from the SAME sampler, scored with the current
    network, and thinned back to the budget with

        p(x) proportional to |r(x)|^k / E[|r|^k] + c

    drawn WITHOUT replacement inside each profile, so the point count per profile is unchanged and
    only the placement moves. Every profile keeps its own budget rather than the whole batch being
    pooled: otherwise one hard input function would eat the allocation of all the others, and the
    operator has to be learned over the whole input distribution, not at its worst member.
    """
    ny = X_pde.shape[1]
    pool = torch.as_tensor(np.asarray(Data.data_pde(Data.N_pde, int(ny * pool_mult))),
                           dtype=torch.float32)
    keep = torch.empty((pool.shape[0], ny, pool.shape[2]), dtype=torch.float32)
    eps_all = []
    for a, b in _chunks(pool.shape[0], chunk):
        with torch.enable_grad():
            r = trainer.pde_residual_abs(pool[a:b].to(device)).cpu()
        eps_all.append(r)
        w = r.double() ** k
        w = w / w.mean(dim=1, keepdim=True).clamp_min(1e-30) + c
        idx = torch.multinomial(w, ny, replacement=False)
        keep[a:b] = torch.gather(pool[a:b], 1, idx.unsqueeze(-1).expand(-1, -1, pool.shape[2]))
    eps = torch.cat(eps_all)
    with torch.enable_grad():                       # what the kept points actually look like
        sel = trainer.pde_residual_abs(keep[:chunk].to(device)).cpu()
    logger.info("ADAPTIVE_SAMPLE pool=%dx k=%.2f c=%.2f | |r| pool: mean %.3e p90 %.3e max %.3e "
                "-> kept: mean %.3e p90 %.3e", pool_mult, k, c,
                eps.mean(), eps.quantile(0.9), eps.max(), sel.mean(), sel.quantile(0.9))
    return keep


def resume_state(trainer, tracker, Data, loaders, resample_every):
    """Put an interrupted run back where its latest checkpoint left it; returns (start_epoch, loaders).

    Weights, Adam moments and the best/early-stopping counters come from the checkpoint. The lr is
    stepped once from the saved value, exactly as the loop does at the end of that epoch. The data are
    not stored, so they are REGENERATED: in the Adam stage the only random draws are the resampling
    of the PI sets (numpy, every resample_every epochs) and the shuffling seeds the DataLoaders draw
    from torch's global generator when an epoch starts iterating them. Replaying both from SEED for
    every epoch up to the checkpoint gives the same data, the same generator states and hence the
    same batches as an uninterrupted run. Checkpoints that carry the generator states are compared
    against the replay, and a mismatch stops the run instead of continuing on different data."""
    ck = tracker.resume_checkpoint()
    epoch = tracker.restore(ck)
    trainer.network.load_state_dict(ck['model_state_dict'])
    trainer.optimizer.load_state_dict(ck['optimizer_state'])
    if trainer.scheduler is not None and (trainer.change_opt or getattr(trainer, 'lr_decay', False)):
        trainer.scheduler.last_epoch = epoch           # chainable form: lr(epoch+1) from the saved lr(epoch)
        trainer.scheduler.step()
    loader_pde, loader_ic, loader_bc, loader_val = loaders
    n_res = 0
    for e in range(epoch + 1):
        if resample_every > 0 and e > 0 and e % resample_every == 0:
            loader_pde, loader_ic, loader_bc = Data.prepare_data_pi()[:3]
            n_res += 1
        for ld in (loader_pde, loader_ic, loader_bc):
            if not isinstance(ld, list):
                next(iter(ld))                         # draws the epoch's shuffling seeds
    rng = ck.get('rng')
    if rng is not None:
        name, keys, pos, has_gauss, cached = rng['numpy']
        ok_t = torch.equal(torch.get_rng_state(), rng['torch'])
        st = np.random.get_state()
        ok_n = (st[0] == name and np.array_equal(st[1], np.asarray(keys, dtype=st[1].dtype))
                and int(st[2]) == pos and int(st[3]) == has_gauss and float(st[4]) == cached)
        if not (ok_t and ok_n):
            raise RuntimeError('resume: the replayed random state differs from the checkpoint (torch %s, '
                               'numpy %s) - the run would continue on different data' % (ok_t, ok_n))
    tracker.logger.info("Resume: lr %.4e from epoch %d | %d resamplings replayed | random state %s",
                        trainer.optimizer.param_groups[0]['lr'], epoch + 1, n_res,
                        'verified against the checkpoint' if rng is not None
                        else 'not stored in this checkpoint (written before the resume support)')
    return epoch + 1, (loader_pde, loader_ic, loader_bc, loader_val)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg_dict = dict(
        electrode='Negative', param_set=SAMPLER_PARAM_SET,        # sampler bookkeeping only
        b1_dim=N_SENSORS, b2_dim=1, hidden_dim=HIDDEN, num_hidden_layers=N_LAYERS,
        Ny_pde=NY_PDE, Ny_bc=NY_BC, Ny_ic=NY_IC, N_pde=N_PDE, N_bc=N_BC, N_ic=N_IC,
        num_batches=NUM_BATCHES, scale_length1=SCALE_LENGTH1, scale_length2=SCALE_LENGTH2,
        scale_factor1=SCALE_FACTOR1, scale_factor2=PHI_SF, device=DEVICE, profile_mode=PROFILE_MODE,
        phi_dist=PHI_DIST, phi_span=PHI_SPAN,
        geometry=GEOMETRY, geometry_law=GEOMETRY_LAW, profile_amp_max=PROFILE_AMP_MAX,
        profile_layer_frac=PROFILE_LAYER_FRAC, profile_layer_delta=PROFILE_LAYER_DELTA,
        xi_map=XI_MAP, xi_warp_p=XI_WARP_P, xi_colloc=XI_COLLOC, profile_space=PROFILE_SPACE,
        t_hi=(min(1.0, 1.15 * TAU_DEPLOY_MAX / TAU_MAX) if T_HI == 'auto' else float(T_HI)),
        tau_split=TAU_SPLIT, tau_density=TAU_DENSITY,
        tau_density_bc=TAU_DENSITY_BC,
        profile_amp_random=PROFILE_AMP_RANDOM, phi_prev_match=PHI_PREV_MATCH,
        shaped_grf_frac=SHAPED_GRF_FRAC, shaped_indep_frac=SHAPED_INDEP_FRAC,
        shaped_m_range=tuple(SHAPED_M_RANGE), shaped_noise=SHAPED_NOISE,
        shaped_depth_max=SHAPED_DEPTH_MAX,
        xi_quad=XI_QUAD,
        t_tr=1.0,                                                  # unused in v2 (kept for Config)
        epochs=EPOCHS, lr=LR, lr_end_factor=LR_END_FACTOR, loss_weights=LOSS_WEIGHTS,
        hard_bc=False,               # the ansatz lives INSIDE MIONetND (q = -Phi); no set_hard_bc
    )
    cfg = Config(**cfg_dict)
    cfg.rollout_K, cfg.w_rollout, cfg.rollout_n_pts = ROLLOUT_K, W_ROLLOUT, ROLLOUT_N_PTS

    logger.info("MIONet v2 (dimensionless) | device=%s | tau_max=%.2f phi_sf=%.2f | sensors=%d h=%d L=%d | "
                "epochs=%d | %s", DEVICE, TAU_MAX, PHI_SF, N_SENSORS, HIDDEN, N_LAYERS, EPOCHS,
                'SMOKE' if SMOKE else 'FULL')

    Data = data(cfg)
    # RESOLVED values, not the 'auto' sentinels: the log has to say what the run actually did.
    logger.info("radial mesh: geometry=%s law=%s | xi_map=%s (p=%.2f) | colloc=%s quad=%s | "
                "t_hi=%.3f tau_split=%s | profile_space=%s | ref_grid=%s | amp_max=%s "
                "| layer=%.2f in %s"
                "\n  sensors=%s",
                GEOMETRY, GEOMETRY_LAW, Data.xi_map, XI_WARP_P, Data.xi_colloc, Data.xi_quad,
                Data.t_hi, Data.tau_split, Data.profile_space, REF_GRID,
                PROFILE_AMP_MAX, PROFILE_LAYER_FRAC,
                tuple(PROFILE_LAYER_DELTA),
                np.array2string(Data.sensors1, precision=3, max_line_width=200))
    X_pde_raw, X_ic_raw, X_bc_raw = Data.data_pde(), Data.data_ic(), Data.data_bc()
    val_pi = Data.data_val_pi(**VAL_PI_SIZES)                      # held-out PI validation
    loader_val = None                              # dummy_val is built once num_batches is known

    model = MIONetND(N_SENSORS, HIDDEN, N_LAYERS, hard_bc=HARD_BC, activation=ACTIVATION,
                     time_features=TIME_FEATURES)
    v2_meta = dict(version='v2-nondim', tau_max=TAU_MAX, phi_sf=PHI_SF, hard_bc=HARD_BC,
                   rollout_K=ROLLOUT_K, w_rollout=W_ROLLOUT, architecture=model.config,
                   geometry=GEOMETRY, geometry_law=GEOMETRY_LAW,
                   xi_map=Data.xi_map, xi_warp_p=XI_WARP_P, xi_colloc=Data.xi_colloc,
                   xi_quad=Data.xi_quad, cons_n_ksi=CONS_N_KSI, cons_n_t=CONS_N_T,
                   profile_space=Data.profile_space, ref_grid=REF_GRID, sensors=Data.sensors1.tolist())
    tracker = TrainingTracker(run_base_dir=RUN_BASE, run_name=RUN_NAME, monitor=MONITOR, mode='min',
                              monitor2=(MONITOR2 if (MONITOR2 and TRACK_PI_VAL) else None), mode2='min',
                              config={'config_dict': cfg_dict, 'v2': v2_meta}, log_every=LOG_EVERY,
                              save_full_model_object=False,          # state_dict payloads only
                              patience=PATIENCE, resume=RESUME,
                              plot_keys=(['loss', 'pde_loss', 'ic_loss', 'bc_loss', 'cons_loss']
                                         + (['pi_val_loss'] if TRACK_PI_VAL else [])))
    trainer = TrainerND(model, cfg, tracker, tau_max=TAU_MAX, val_pi=val_pi, lr_lbfgs=LR_LBFGS,
                        lbfgs_full_batch=LBFGS_FULL_BATCH, lbfgs_ny=(LBFGS_NY_PDE, LBFGS_NY_BC, LBFGS_NY_IC),
                        pde_form=PDE_FORM, lbfgs_resample=LBFGS_RESAMPLE)
    trainer.change_opt = LBFGS_PHASE
    trainer.lr_decay = LR_DECAY
    if LR_DECAY and LR_DECAY_EPOCHS:
        from torch.optim.lr_scheduler import LinearLR
        trainer.scheduler = LinearLR(trainer.optimizer, start_factor=1.0, end_factor=LR_END_FACTOR,
                                     total_iters=int(LR_DECAY_EPOCHS))
    if ES_FROM_EPOCH:
        _es = tracker.early_stop_check
        tracker.early_stop_check = lambda *, epoch: epoch >= ES_FROM_EPOCH and _es(epoch=epoch)
    trainer.xi_quad = Data.xi_quad     # the RESOLVED law, not the raw 'follow' sentinel
    trainer.xi_warp_p = XI_WARP_P      # only read when xi_quad == 'warp'
    trainer.cons_n_ksi, trainer.cons_n_t = CONS_N_KSI, CONS_N_T
    trainer.cons_use_ic = CONS_USE_IC
    trainer.cons_legacy = CONS_LEGACY
    trainer.rollout_n_profiles = ROLLOUT_N_PROFILES
    trainer.lbfgs_patience = LBFGS_PATIENCE
    trainer.track_pi_val = TRACK_PI_VAL

    if INIT_FROM:
        load_weights(model, INIT_FROM)
    if ADAPTIVE_SAMPLE:
        if not INIT_FROM:
            logger.warning("ADAPTIVE_SAMPLE without INIT_FROM: the residual map of an untrained "
                           "network is noise, and the sampler will chase it")
        X_pde_raw = adaptive_resample(trainer, Data, X_pde_raw, *ADAPTIVE_SAMPLE, device=DEVICE)

    # Batch size is chosen AFTER the model and the validation reference are on the card, so the
    # free-memory measurement reflects what is actually left for the training step.
    nb = NUM_BATCHES
    if str(NUM_BATCHES).lower() == 'auto':
        nb = autotune_num_batches(trainer, X_pde_raw, X_ic_raw, X_bc_raw,
                                  n_batches_hint=NUM_BATCHES_HINT, device=DEVICE)
    Data.num_batches = cfg.num_batches = cfg_dict['num_batches'] = int(nb)
    loader_pde = create_loader(X_pde_raw, int(nb))
    loader_ic = create_loader(X_ic_raw, int(nb))
    loader_bc = create_loader(X_bc_raw, int(nb))
    dummy_val = [torch.zeros((1, 1, 2 + N_SENSORS + 2), dtype=torch.float32)] * int(nb)
    if loader_val is None:                          # PI-only: the dummy loader must match num_batches
        loader_val = dummy_val
    logger.info("batches: %d  (%d pde / %d ic / %d bc profiles per batch)", int(nb),
                -(-X_pde_raw.shape[0] // int(nb)), -(-X_ic_raw.shape[0] // int(nb)),
                -(-X_bc_raw.shape[0] // int(nb)))
    # the resolved count goes into training.log too: with 'auto' it is the only record of it, and a
    # resume has to use the same one (read BEFORE this process adds its own line)
    nb_logged = re.findall(r'batches per epoch: (\d+) \(num_batches=', open(tracker.log_path).read()) \
        if RESUME and tracker.log_path is not None and tracker.log_path.exists() else []
    tracker.logger.info("batches per epoch: %d (num_batches=%s)", int(nb), NUM_BATCHES)
    start_epoch = 0
    if RESUME:
        assert not (ONLY_LBFGS or LBFGS_PHASE or INIT_FROM or ADAPTIVE_SAMPLE), 'resume covers the Adam stage only'
        saved = json.load(open(os.path.join(tracker.run_dir, 'config.json')))['config_dict']
        diff = {k: (saved.get(k), v) for k, v in json.loads(json.dumps(cfg_dict, default=str)).items()
                if saved.get(k) != v}
        if str(saved.get('num_batches')).lower() == 'auto' and str(NUM_BATCHES).lower() == 'auto':
            diff.pop('num_batches', None)          # config.json keeps 'auto'; compare the resolved count
            if nb_logged and int(nb_logged[0]) != int(nb):
                diff['num_batches'] = (int(nb_logged[0]), int(nb))
            elif not nb_logged:
                tracker.logger.warning("resume: the original run did not record its resolved batch count; "
                                       "autotune chose %d again", int(nb))
        if diff:                                   # device included: never continue a GPU run on the CPU
            raise SystemExit('resume: settings differ from the original run %s: %s' % (tracker.run_dir, diff))
        start_epoch, (loader_pde, loader_ic, loader_bc, loader_val) = resume_state(
            trainer, tracker, Data, (loader_pde, loader_ic, loader_bc, loader_val), RESAMPLE_EVERY)
    trainer.train(loader_pde, loader_ic, loader_bc, loader_val, only_LBFGS=ONLY_LBFGS,
                  lr_lbfgs=LR_LBFGS,
                  resample_fn=(lambda: Data.prepare_data_pi()[:3]) if RESAMPLE_EVERY > 0 else None,
                  resample_every=RESAMPLE_EVERY, start_epoch=start_epoch)

    # ---------------------------- self-describing artifact ----------------------------
    train_meta = dict(epochs=EPOCHS, loss_weights=LOSS_WEIGHTS, rollout_K=ROLLOUT_K, lr=LR, activation=ACTIVATION,
                      hpo_config=HPO_CONFIG, time_features=TIME_FEATURES, lbfgs_phase=LBFGS_PHASE, lr_lbfgs=LR_LBFGS,
                      lbfgs_full_batch=LBFGS_FULL_BATCH, lbfgs_ny=[LBFGS_NY_PDE, LBFGS_NY_BC, LBFGS_NY_IC],
                      pde_form=PDE_FORM, lbfgs_resample=LBFGS_RESAMPLE,
                      rollout_n_profiles=ROLLOUT_N_PROFILES, cons_use_ic=CONS_USE_IC,
                      resample_every=RESAMPLE_EVERY,
                      geometry=GEOMETRY, geometry_law=GEOMETRY_LAW,
                      # the SAMPLER, so a later run can reproduce this one from the
                      # artifact alone. t_hi was missing until 2026-09-25 and
                      # refine_lbfgs silently read the default instead - see the model card
                      t_hi=Data.t_hi, tau_split=TAU_SPLIT,
                      profile_amp_random=PROFILE_AMP_RANDOM, phi_prev_match=PHI_PREV_MATCH,
                      scale_length1=SCALE_LENGTH1,
                      xi_map=Data.xi_map, xi_warp_p=XI_WARP_P, xi_colloc=Data.xi_colloc,
                      xi_quad=Data.xi_quad, cons_n_ksi=CONS_N_KSI, cons_n_t=CONS_N_T,
                      lbfgs_patience=LBFGS_PATIENCE, monitor2=MONITOR2,
                      profile_space=Data.profile_space, ref_grid=REF_GRID, ref_r_pts=REF_R_PTS,
                      track_pi_val=TRACK_PI_VAL, monitor=MONITOR, lr_decay=LR_DECAY, lr_decay_epochs=LR_DECAY_EPOCHS,
                      es_from_epoch=ES_FROM_EPOCH, patience=PATIENCE,
                      val_pi_sizes=(VAL_PI_SIZES if TRACK_PI_VAL else None),
                      profile_mode=PROFILE_MODE, scale_factor1=SCALE_FACTOR1,
                      shaped_grf_frac=SHAPED_GRF_FRAC, shaped_indep_frac=SHAPED_INDEP_FRAC,
                      shaped_m_range=list(SHAPED_M_RANGE), shaped_noise=SHAPED_NOISE,
                      shaped_depth_max=SHAPED_DEPTH_MAX,
                      profile_amp_max=PROFILE_AMP_MAX, profile_layer_frac=PROFILE_LAYER_FRAC,
                      cons_legacy=CONS_LEGACY,
                      profile_layer_delta=list(PROFILE_LAYER_DELTA),
                      N_pde=N_PDE, Ny_pde=NY_PDE, N_ic=N_IC, N_bc=N_BC, seed=SEED, smoke=SMOKE)
    write_artifact(str(tracker.run_dir), model, train_meta, tau_max=TAU_MAX, phi_sf=PHI_SF, device=DEVICE,
                   sensors=Data.sensors1)


def write_artifact(run_dir, model, train_meta, tau_max, phi_sf, device, finalized=False, sensors=None):
    """model.pt (best state_dict ONLY) + surrogate_manifest.json + metadata.json + history.csv from
    checkpoints/best.pt. Also used by `train_nd.py --finalize RUN_DIR` for a run whose process was
    stopped before this block ran (the tracker keeps best.pt at every improvement)."""
    best = torch.load(os.path.join(run_dir, 'checkpoints', 'best.pt'), map_location='cpu', weights_only=False)
    sd = best['model_state_dict']
    torch.save(sd, os.path.join(run_dir, 'model.pt'))                 # weights ONLY
    manifest = dict(
        version='v2-nondim', architecture=model.config,
        sensors=(sensor_grid(model.n_sensors) if sensors is None else np.asarray(sensors)).tolist(),
        dimensionless=dict(tau_range=[0.0, tau_max], phi_sampling=f'N(0,{phi_sf})',
                           phi_range_recommended=[-2.0 * phi_sf, 2.0 * phi_sf],
                           time_column='tau/tau_max in [0,1]', state='x = c/c_max, centre-pinned'),
        chemistry_agnostic=True, electrodes='both (sign carried by Phi)',
        pde='dx/dtau = xi^-2 d/dxi(xi^2 dx/dxi); -dx/dxi|1 = Phi; hard-BC q=-Phi',
        training=dict(best_epoch=int(best.get('epoch', best.get('best_epoch', -1))),
                      best_val_loss=float(best.get('best_value', float('nan'))),
                      finalized_from_checkpoint=finalized, **train_meta),
        weights_file='model.pt', torch=torch.__version__,
    )
    with open(os.path.join(run_dir, 'surrogate_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(run_dir, 'metadata.json'), 'w') as f:
        json.dump(dict(date=datetime.datetime.now().isoformat(timespec='seconds'),
                       description='MIONet v2: dimensionless single-particle diffusion operator, '
                                   'physics-informed only (PDE/IC/BC/cons), one net for both electrodes'
                                   + (' [finalized from checkpoints/best.pt after the process was stopped]'
                                      if finalized else ''),
                       dataset='PI collocation in (xi, tau) via srcs.prepData; validation on a held-out PI '
                               'collocation set (no labelled data)',
                       git_hash=LAUNCH_HASH, git_hash_at_save=_git_hash(), device=device, smoke=bool(train_meta.get('smoke', False))), f, indent=2)
    # A second, independently loadable artifact from the best MONITOR2 checkpoint. Same folder
    # layout, so Surrogate.from_run(<run>/best_pi) and srcs/evaluation.py --models "<run>/best_pi"
    # both work without knowing anything about it.
    for ck in sorted(f for f in os.listdir(os.path.join(run_dir, 'checkpoints'))
                     if f.startswith('best_') and f.endswith('.pt')):
        sub = os.path.join(run_dir, 'best_pi')
        os.makedirs(sub, exist_ok=True)
        b2 = torch.load(os.path.join(run_dir, 'checkpoints', ck), map_location='cpu', weights_only=False)
        torch.save(b2['model_state_dict'], os.path.join(sub, 'model.pt'))
        m2 = dict(manifest)
        m2['training'] = dict(manifest['training'], selected_on=ck[5:-3],
                              best_epoch=int(b2.get('epoch', -1)),
                              best_value=float(b2.get('metrics', {}).get(ck[5:-3], float('nan'))))
        with open(os.path.join(sub, 'surrogate_manifest.json'), 'w') as f:
            json.dump(m2, f, indent=2)
        logger.info("ARTIFACT %s : selected on %s, epoch %s", sub, ck[5:-3], b2.get('epoch'))
        break
    if os.path.exists(os.path.join(run_dir, 'metrics.csv')):
        shutil.copyfile(os.path.join(run_dir, 'metrics.csv'), os.path.join(run_dir, 'history.csv'))
    logger.info("ARTIFACT %s : model.pt + surrogate_manifest.json + metadata.json + history.csv | "
                "best epoch %s val %.3e", run_dir, best.get('epoch'), best.get('best_value', float('nan')))


def finalize(run_dir):
    """`train_nd.py --finalize RUN_DIR`: write the artifact for a run whose process was stopped
    (e.g. killed during the L-BFGS phase). Hyper-parameters are read from the run's config.json."""
    cfg_all = json.load(open(os.path.join(run_dir, 'config.json')))
    cd, v2 = cfg_all.get('config_dict', {}), cfg_all.get('v2', {})
    arch = v2.get('architecture') or dict(n_sensors=cd['b1_dim'], hidden_dim=cd['hidden_dim'],
                                          n_hidden_layers=cd['num_hidden_layers'],
                                          hard_bc=True, activation='tanh')
    model = MIONetND.from_config(arch)
    tau_max, phi_sf = float(v2.get('tau_max', TAU_MAX)), float(v2.get('phi_sf', cd.get('scale_factor2', PHI_SF)))
    sensors = (np.asarray(v2['sensors']) if v2.get('sensors') else
               sensor_grid(arch['n_sensors'], cd.get('xi_map', 'uniform'), cd.get('xi_warp_p', 2.0)))
    train_meta = dict(epochs=cd.get('epochs'), loss_weights=cd.get('loss_weights'), rollout_K=v2.get('rollout_K'),
                      lr=cd.get('lr'), activation=arch.get('activation'),
                      profile_mode=cd.get('profile_mode', 'grf'), scale_factor1=cd.get('scale_factor1'),
                      N_pde=cd.get('N_pde'), Ny_pde=cd.get('Ny_pde'),
                      N_ic=cd.get('N_ic'), N_bc=cd.get('N_bc'), smoke=False,
                      note='finalized from checkpoints/best.pt; hyper-parameters read from config.json')
    write_artifact(run_dir, model, train_meta, tau_max=tau_max, phi_sf=phi_sf, device='cpu', finalized=True,
                   sensors=sensors)

if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--finalize':
        finalize(os.path.abspath(sys.argv[2]))
    else:
        main()
