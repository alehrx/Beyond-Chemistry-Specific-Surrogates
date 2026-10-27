"""
hpo_rollout.py
==============
Optuna search for the parameter-free MIONet with a CHEMISTRY-FREE objective: the error of
multi-step roll-outs against the exact solution of the dimensionless particle problem
(srcs_EM.exact). No PyBaMM, no parameter set: the reference is the training equation itself.

Example (an earlier roll-out study, ~4 h on an RTX 4070; a study config can also run the search as
its first stage, "hyperparameters": {"mode": "run", ...}):

    python srcs/hpo_rollout.py --name hpo_rollout_s10_p2_noproj --steps 10 --epochs 250 \
        --hidden 16 32 64 128 --layers 2 8 --lr-range 1e-4 3e-3 --pilot-mult 2 --batches 16 \
        --no-project --with-dd

The objective
  128 chains x --steps steps (10 in the reported study), fixed for the whole study (own RNG, seed 2024):
    x0     drawn from the training profile sampler (centre-pinned random field + parabola)
    Phi    piecewise constant in segments of geometric length (mean 8 steps), N(0, 1.5^2), |Phi|<=3
    dtau   per step, log-uniform in [0.002, 0.03] - the range a 20-s step / the adaptive window
           spans over the five evaluation cells
  The network advances each chain on its 21 sensors exactly as at deployment (centre-pinned
  input); the reference advances the piecewise-linear initial profile exactly. Objective = mean
  over chains and steps of the MSE at the sensors. With --no-project (the reported study) the
  lithium-balance projection is NOT applied inside the objective: the projection would hide the
  drift that the mass-conservation loss weight controls. Diagnostics stored per trial: the error
  with and without the projection, the surface-only error, and (with --with-dd) the one-step
  PyBaMM objective of runs/hpo_study, so the two objectives can be compared.

Everything else follows runs/hpo_study: the m21d training configuration, the pilot PI sampler
(here with twice the input functions, cache key 3cfd1fdf6cb09861), TPE seed 42, median pruner,
60 trials. The best trial is written to model_config.json, which scripts/run_training.py trains in full
("hyperparameters": {"mode": "file", "path": <folder>}).
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPT_DIR)
log = logging.getLogger(__name__)

N_CHAINS, N_STEPS, CHAIN_SEED = 128, 60, 2024
DTAU_RANGE, SEG_MEAN, PHI_CLIP = (0.002, 0.03), 8, 3.0


def force_m21d(T):
    """The reported training configuration, set on train_nd BEFORE hpo_nd imports from it
    (as scripts/run_training.py does for the full training)."""
    T.N_SENSORS = 21
    T.TAU_MAX, T.PHI_SF = 0.2, 1.5
    T.PHI_DIST, T.PHI_SPAN = 'normal', 2.0
    T.PROFILE_MODE = 'grf+parabola'
    T.SCALE_LENGTH1, T.SCALE_FACTOR1 = [0.5, 1.0], 0.2
    T.SCALE_LENGTH2 = [0.1, 1.0]
    T.PROFILE_AMP_MAX, T.PROFILE_LAYER_FRAC = None, 0.0
    T.PROFILE_AMP_RANDOM, T.PHI_PREV_MATCH = False, 0.5
    T.GEOMETRY, T.XI_MAP, T.XI_QUAD = 'uniform', 'uniform', 'uniform'
    T.XI_COLLOC, T.T_HI = 'uniform', 1.0
    T.TAU_DENSITY, T.TAU_SPLIT = (0.0,), None
    T.TAU_DENSITY_BC = True
    T.PROFILE_SPACE, T.REF_GRID, T.REF_R_PTS = 'physical', 'fv', 32
    T.PDE_FORM = 'conservative'
    T.CONS_N_KSI, T.CONS_N_T, T.CONS_USE_IC = 41, 4, False
    T.CONS_LEGACY = False
    T.TIME_FEATURES = 'linear'


def search_space(T):
    """The search space, pilot-set sizes and trial trainer (formerly in hpo_nd.py, the one-step study of
    runs/hpo_study, whose pilot sampler, ranges and pruning interval this study keeps)."""
    from types import SimpleNamespace
    from srcs.training import Trainer
    from srcs_EM.reference import stratified_mse

    class TrainerHPO(T.TrainerND):
        """TrainerND residuals; loss_dd is replaced by the roll-out objective in main(). The stratified
        one-step PyBaMM error is kept for the --with-dd diagnostic (its groups are held here: TrainerND
        itself no longer carries any labelled reference)."""
        def __init__(self, *args, val_groups=None, **kw):
            super().__init__(*args, **kw)
            self.val_groups = val_groups

        def loss_dd(self, XX):
            if self.val_groups is None or XX.shape[0] != self.val_groups.shape[0]:
                return Trainer.loss_dd(self, XX)
            n_s = XX.shape[1]
            e = self.network(XX[..., :1], XX[..., 1:2], XX[..., 2:2 + n_s], XX[..., 2 + n_s:3 + n_s]) - XX[..., -1:]
            strat, _, _ = stratified_mse(e.pow(2).mean(dim=(1, 2)), self.val_groups)
            return torch.tensor(strat, device=XX.device)

    return SimpleNamespace(
        DATA=dict(N_pde=150, Ny_pde=3_000, N_ic=200, Ny_ic=100, N_bc=200, Ny_bc=300, num_batches=4),
        VAL=dict(N_pde=20, Ny_pde=500, N_ic=50, Ny_ic=50, N_bc=50, Ny_bc=100),
        LR_RANGE=(1e-4, 1e-3), W_PDE_RANGE=(1.0, 3e2), W_IC_RANGE=(5e2, 5e4), W_BC_RANGE=(1.0, 3e2),
        W_CONS_RANGE=(0.1, 1e2), ACTIVATIONS=['tanh', 'silu', 'gelu'], PRUNE_INTERVAL=10,
        TrainerHPO=TrainerHPO)


class RolloutSet:
    """Fixed roll-out chains and their exact references, and the network's error on them."""

    def __init__(self, x0_sensors, sensors, tau_max, rng):
        from srcs_EM.exact import SphereExact
        self.S = SphereExact()
        self.s, self.tau_max = np.asarray(sensors, float), float(tau_max)
        C = x0_sensors.shape[0]
        self.phi = np.zeros((C, N_STEPS))
        for c in range(C):
            k = 0
            while k < N_STEPS:
                L = 1 + rng.geometric(1.0 / SEG_MEAN)
                self.phi[c, k:k + L] = np.clip(rng.normal(0.0, 1.5), -PHI_CLIP, PHI_CLIP)
                k += L
        lo, hi = np.log(DTAU_RANGE[0]), np.log(DTAU_RANGE[1])
        self.dtau = np.exp(rng.uniform(lo, hi, (C, N_STEPS)))
        self.x0 = np.asarray(x0_sensors, float)
        # exact reference at the sensors after every step
        xf = self.S.to_fine(self.x0, self.s)
        ref = []
        for k in range(N_STEPS):
            xf = self.S.step(xf, self.phi[:, k], self.dtau[:, k])
            ref.append(self.S.at(xf, self.s))
        self.ref = np.stack(ref, axis=1)                        # (C, K, n_sensors)
        # volume-mean weights of the piecewise-linear profile through the sensor values
        hats = self.S.to_fine(np.eye(len(self.s)), self.s)
        self.wv = 3.0 * (hats @ self.S.wq)                      # mean = x @ wv

    @torch.no_grad()
    def evaluate(self, net, device, project=True):
        n = len(self.s)
        C = self.x0.shape[0]
        r = torch.as_tensor(self.s, dtype=torch.float32, device=device).view(1, n, 1).expand(C, n, 1)
        wv = torch.as_tensor(self.wv, dtype=torch.float32, device=device)
        x = torch.as_tensor(self.x0, dtype=torch.float32, device=device)
        phi = torch.as_tensor(self.phi, dtype=torch.float32, device=device)
        dtau = torch.as_tensor(self.dtau, dtype=torch.float32, device=device)
        out = []
        for k in range(N_STEPS):
            pin = x[:, :1]
            t = (dtau[:, k] / self.tau_max).view(C, 1, 1).expand(C, n, 1)
            y = net(t, r, (x - pin).unsqueeze(1), phi[:, k].view(C, 1, 1)).squeeze(-1) + pin
            if project:
                target = x @ wv - 3.0 * phi[:, k] * dtau[:, k]
                y = y + (target - y @ wv).unsqueeze(1)
            x = y
            out.append(x)
        pred = torch.stack(out, 1).cpu().numpy()
        err = (pred - self.ref) ** 2
        return dict(mse=float(err.mean()), mse_surf=float(err[..., -1].mean()),
                    mse_last=float(err[:, -1].mean()))


def main():
    global N_STEPS
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='hpo_rollout')
    ap.add_argument('--hidden', nargs='+', type=int, default=[16, 32, 64, 128])
    ap.add_argument('--layers', nargs=2, type=int, default=[2, 8])
    ap.add_argument('--lr', type=float, default=None, help='fixed lr (searched if omitted)')
    ap.add_argument('--trials', type=int, default=60)
    ap.add_argument('--steps', type=int, default=N_STEPS, help='roll-out length of every chain')
    ap.add_argument('--lr-range', nargs=2, type=float, default=None, metavar=('LO', 'HI'),
                    help='learning-rate search range (default: hpo_nd.LR_RANGE)')
    ap.add_argument('--pilot-mult', type=int, default=1,
                    help='multiply the number of pilot input functions (PDE, IC, BC)')
    ap.add_argument('--batches', type=int, default=None, help='mini-batches per epoch (default 4)')
    ap.add_argument('--no-project', action='store_true',
                    help='score the roll-out WITHOUT the mass-conservation projection, i.e. the raw '
                         'network drift; the projection hides exactly what the MC loss controls')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--torch-seed', type=int, default=42)
    ap.add_argument('--with-dd', action='store_true', help='also record the one-step PyBaMM objective')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--key-only', action='store_true', help='print the pilot-data cache key and stop')
    ap.add_argument('--no-hard-bc', dest='hard_bc', action='store_false',
                    help='drop the -Phi r^2/2 term from the network output (on by default)')
    ap.add_argument('--eval-runs', nargs='+', default=None,
                    help='score trained run folders (model.pt + config.json) on the roll-out set and stop')
    a = ap.parse_args()
    N_STEPS = a.steps
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    import train_nd as T
    force_m21d(T)
    H = search_space(T)
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    import srcs.hpo_common as HC
    from srcs.prepData import Config, data, create_loader
    from srcs_EM import MIONetND

    epochs, trials = (2, 2) if a.smoke else (a.epochs, a.trials)
    DATA = dict(N_pde=6, Ny_pde=200, N_ic=6, Ny_ic=20, N_bc=6, Ny_bc=30, num_batches=1) if a.smoke else dict(H.DATA)
    for k in ('N_pde', 'N_ic', 'N_bc'):
        DATA[k] *= a.pilot_mult
    if a.batches:
        DATA['num_batches'] = a.batches
    lr_range = tuple(a.lr_range) if a.lr_range else H.LR_RANGE
    log.info('pilot set %s | lr range %s', DATA, lr_range)
    VAL = dict(N_pde=3, Ny_pde=100, N_ic=5, Ny_ic=20, N_bc=5, Ny_bc=30) if a.smoke else H.VAL
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if a.key_only or a.eval_runs:
        out_dir = None
    else:
        out_dir = str(HC.make_hpo_dir(a.name + ('_smoke' if a.smoke else ''), base=os.path.join(ROOT, 'runs')))

    data_cfg = dict(electrode='Negative', param_set=T.SAMPLER_PARAM_SET, b1_dim=T.N_SENSORS, b2_dim=1, **DATA,
                    scale_length1=T.SCALE_LENGTH1, scale_length2=T.SCALE_LENGTH2,
                    scale_factor1=T.SCALE_FACTOR1, profile_mode=T.PROFILE_MODE,
                    geometry=T.GEOMETRY, geometry_law=T.GEOMETRY_LAW, xi_warp_p=T.XI_WARP_P,
                    profile_amp_max=T.PROFILE_AMP_MAX, profile_layer_frac=T.PROFILE_LAYER_FRAC,
                    profile_layer_delta=T.PROFILE_LAYER_DELTA,
                    phi_dist=T.PHI_DIST, phi_span=T.PHI_SPAN, t_hi=float(T.T_HI),
                    xi_map=T.XI_MAP, xi_colloc=T.XI_COLLOC, xi_quad=T.XI_QUAD,
                    tau_density=T.TAU_DENSITY, tau_split=T.TAU_SPLIT,
                    profile_space=T.PROFILE_SPACE, profile_amp_random=T.PROFILE_AMP_RANDOM,
                    phi_prev_match=T.PHI_PREV_MATCH,
                    scale_factor2=T.PHI_SF, device=device, t_tr=1.0, epochs=epochs, lr=5e-4,
                    lr_end_factor=1e-2, loss_weights=[1.0, 1e3, 10.0, 10.0], hidden_dim=32,
                    num_hidden_layers=3, hard_bc=False)
    log.info('pilot PI data key %s (hpo_study: bd2d1b4c05acf7a0)', HC._cache_key(data_cfg, VAL, 42))
    if a.key_only:
        return
    (X_pde, X_ic, X_bc), val_pi = HC.get_pi_dataset(data_cfg, VAL, T.CACHE_DIR)
    pi_data = data(Config(**data_cfg))

    # the roll-out set: its own RNG for the chains; the profile sampler uses numpy's global RNG,
    # whose state is saved and restored so nothing else in the study shifts
    state = np.random.get_state()
    np.random.seed(CHAIN_SEED)
    x0 = pi_data.data_ic(N_CHAINS, 2)[:, 0, 2:2 + T.N_SENSORS].numpy().astype(float)
    np.random.set_state(state)
    R = RolloutSet(x0, pi_data.sensors1, T.TAU_MAX, np.random.default_rng(CHAIN_SEED))
    log.info('roll-out set: %d chains x %d steps, |Phi| up to %.2f, dtau %.3g-%.3g',
             N_CHAINS, N_STEPS, np.abs(R.phi).max(), R.dtau.min(), R.dtau.max())

    if a.eval_runs:
        for run in a.eval_runs:
            cfg = json.load(open(os.path.join(run, 'config.json')))
            arch = cfg.get('v2', {}).get('architecture', {})
            net = MIONetND(T.N_SENSORS, int(arch['hidden_dim']), int(arch['n_hidden_layers']),
                           hard_bc=bool(arch.get('hard_bc', True)),
                           activation=arch['activation']).to(device)
            sd = torch.load(os.path.join(run, 'model.pt'), map_location=device, weights_only=False)
            net.load_state_dict(sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd)
            net.eval()
            on, off = R.evaluate(net, device), R.evaluate(net, device, project=False)
            log.info('%-40s roll-out MSE %.3e (RMS %.4f) | surface %.3e | last step %.3e | no projection %.3e',
                     os.path.basename(os.path.normpath(run)),on['mse'], on['mse'] ** 0.5, on['mse_surf'],
                     on['mse_last'], off['mse'])
        return
    X_dd = groups = None
    if a.with_dd:
        X_dd, groups = HC.build_val_reference(T, with_groups=True, sensors=pi_data.sensors1, grid=T.REF_GRID)
        X_dd, groups = X_dd.to(device), groups.to(device)

    project = not a.no_project
    log.info('objective: roll-out MSE %s the mass-conservation projection', 'with' if project else 'WITHOUT')
    HC.dd_val_loss = lambda trainer, _X, dev: R.evaluate(trainer.network, dev, project=project)['mse']   # the objective
    loaders = (create_loader(X_pde, DATA['num_batches']), create_loader(X_ic, DATA['num_batches']),
               create_loader(X_bc, DATA['num_batches']))
    torch.manual_seed(a.torch_seed)

    def objective(trial):
        hidden = trial.suggest_categorical('hidden_dim', list(a.hidden))
        layers = trial.suggest_int('num_hidden_layers', *a.layers)
        act = trial.suggest_categorical('activation', H.ACTIVATIONS)
        lr = a.lr if a.lr is not None else trial.suggest_float('lr', *lr_range, log=True)
        w = [trial.suggest_float(k, *rng_, log=True) for k, rng_ in
             (('w_pde', H.W_PDE_RANGE), ('w_ic', H.W_IC_RANGE), ('w_bc', H.W_BC_RANGE), ('w_cons', H.W_CONS_RANGE))]
        model = MIONetND(T.N_SENSORS, hidden, layers, hard_bc=a.hard_bc, activation=act).to(device)
        cfg = Config(**{**data_cfg, 'lr': lr, 'epochs': epochs, 'hidden_dim': hidden,
                        'num_hidden_layers': layers, 'loss_weights': w})
        cfg.rollout_K, cfg.w_dd = 0, 0.0
        tr = H.TrainerHPO(model, cfg, None, tau_max=T.TAU_MAX, val_pi=val_pi, pde_form=T.PDE_FORM,
                          val_groups=groups)
        tr.change_opt = False
        tr.xi_quad, tr.xi_warp_p = pi_data.xi_quad, T.XI_WARP_P
        tr.cons_n_ksi, tr.cons_n_t = T.CONS_N_KSI, T.CONS_N_T
        try:
            val = HC.hpo_train(model, tr, lr, *loaders, None, trial, epochs, H.PRUNE_INTERVAL,
                               data_cfg['lr_end_factor'], device)
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            log.warning('trial %d raised: %s', trial.number, exc)
            return float('inf')
        model.eval()
        for k, v in R.evaluate(model, device).items():
            trial.set_user_attr('rollout_' + k, v)
        trial.set_user_attr('rollout_mse_noproj', R.evaluate(model, device, project=False)['mse'])
        trial.set_user_attr('rollout_mse_proj', R.evaluate(model, device, project=True)['mse'])
        trial.set_user_attr('n_params', sum(p.numel() for p in model.parameters()))
        if X_dd is not None:
            trial.set_user_attr('dd_onestep', float(tr.loss_dd(X_dd)))
        return val

    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42),
                                pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=80,
                                                    interval_steps=H.PRUNE_INTERVAL),
                                study_name='MIONet_ND_HPO_rollout',
                                storage='sqlite:///' + os.path.join(out_dir, 'study.db').replace('\\', '/'))
    study.optimize(objective, n_trials=trials, show_progress_bar=not a.smoke)

    b = study.best_trial
    bp = b.params
    json.dump({'model': 'MIONetND (v2, dimensionless)', 'n_sensors': T.N_SENSORS, 'hard_bc': a.hard_bc,
               'hidden_dim': bp['hidden_dim'],
               'number_of_hidden_layers': bp['num_hidden_layers'], 'activation': bp['activation'],
               'lr': bp.get('lr', a.lr), 'loss_weights': [bp['w_pde'], bp['w_ic'], bp['w_bc'], bp['w_cons']],
               'tau_max': T.TAU_MAX, 'phi_sf': T.PHI_SF, 'pde_form': T.PDE_FORM, 'geometry': T.GEOMETRY,
               'phi_dist': T.PHI_DIST, 'sensors': list(map(float, pi_data.sensors1)),
               'objective': 'roll-out MSE vs the exact dimensionless solution (%d chains x %d steps), %s projection'
                            % (N_CHAINS, N_STEPS, 'with' if project else 'without'), 'rollout_mse': b.value},
              open(os.path.join(out_dir, 'model_config.json'), 'w'), indent=2)
    json.dump({'date': __import__('datetime').datetime.now().isoformat(timespec='seconds'),
               'description': 'HPO with a chemistry-free roll-out objective against the exact solution',
               'dataset': 'pilot PI set (hpo_study data) + exact roll-out chains, no PyBaMM in the objective',
               'args': vars(a), 'pilot': DATA, 'lr_range': list(lr_range), 'n_chains': N_CHAINS, 'n_steps': N_STEPS, 'chain_seed': CHAIN_SEED,
               'dtau_range': DTAU_RANGE, 'phi_segment_mean': SEG_MEAN, 'phi_clip': PHI_CLIP,
               'n_trials': len(study.trials), 'hpo_epochs': epochs},
              open(os.path.join(out_dir, 'metadata.json'), 'w'), indent=2)
    log.info('best trial %d: roll-out MSE %.3e | %s', b.number, b.value, bp)


if __name__ == '__main__':
    main()
