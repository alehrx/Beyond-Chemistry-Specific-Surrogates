"""
run_training.py
===============
One entry point for a training study, configured by one JSON file:

    python scripts/run_training.py configs/<study>.json             # run (finished stages are skipped)
    python scripts/run_training.py configs/<study>.json --resume    # also continue an interrupted seed
    python scripts/run_training.py configs/<study>.json --dry-run   # print the plan, run nothing
    python scripts/run_training.py configs/<study>.json --stages post   # only the post-processing

Stages, in order (each one skipped when its output already exists):

    hpo      hyperparameters: an existing HPO folder, values set by hand, or a new roll-out HPO
    adam     physics-informed Adam stage, one run per seed        -> runs/v2_nondim/<name>_s<seed>
    lbfgs    L-BFGS refinement of each Adam run                   -> runs/v2_nondim/<name>_s<seed>_lbfgs
    score    roll-outs against PyBaMM, full-profile error         -> runs/studies/<name>/results.csv
    post     figures: training curves per seed, HPO, error bins over the seeds
                                                                  -> runs/studies/<name>/figures/

Everything about the study (a copy of the JSON, metadata, the log, the scores, the figures) is in
runs/studies/<name>/. Run folders are never overwritten; an Adam folder without model.pt means an
interrupted run and stops the study with a message, unless --resume is given: then the Adam stage
continues in that folder from its latest checkpoint, exactly as if it had not stopped (see
train_nd.resume_state), and an incomplete L-BFGS folder is renamed <name>_lbfgs_interrupted[N] and the
refinement redone (it is short and seeded). scripts/supervise_training.py relaunches with --resume
after a crash. See configs/README.md for every key.
"""
import argparse
import datetime
import json
import logging
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
SCRIPTS = os.path.join(ROOT, 'scripts')
SRCS = os.path.join(ROOT, 'srcs')
RUNS = os.path.join(ROOT, 'runs', 'v2_nondim')
PY = sys.executable
STAGES = ('hpo', 'adam', 'lbfgs', 'score', 'post')
log = logging.getLogger('run_training')

# JSON key -> train_nd global, for the dataset and the Adam stage
DATASET_KEYS = dict(n_pde='N_PDE', ny_pde='NY_PDE', n_ic='N_IC', ny_ic='NY_IC', n_bc='N_BC', ny_bc='NY_BC',
                    resample_every='RESAMPLE_EVERY')
ADAM_KEYS = dict(epochs='EPOCHS', lr_decay='LR_DECAY', lr_decay_epochs='LR_DECAY_EPOCHS',
                 lr_end_factor='LR_END_FACTOR', patience='PATIENCE', es_from_epoch='ES_FROM_EPOCH',
                 monitor='MONITOR', monitor2='MONITOR2')

# The fixed part of the recipe (train_nd globals): sensors, time and flux domains, how the input
# functions and collocation points are sampled, the form of the losses and the validation sets. These
# are the settings of the paper's networks; a study changes them only through its "recipe" section.
# The JSON sections (dataset, training) and the hyperparameters are applied on top of them.
RECIPE = dict(
    N_SENSORS=21, TAU_MAX=0.2, PHI_SF=1.5, PHI_DIST='normal', PHI_SPAN=2.0,
    PROFILE_MODE='grf+parabola', SCALE_LENGTH1=[0.5, 1.0], SCALE_FACTOR1=0.2,
    SCALE_LENGTH2=[0.1, 1.0], SCALE_FACTOR2=1.5, PROFILE_AMP_MAX=None, PROFILE_LAYER_FRAC=0.0,
    PROFILE_AMP_RANDOM=False, PHI_PREV_MATCH=0.5,
    GEOMETRY='uniform', XI_MAP='uniform', XI_QUAD='uniform', XI_COLLOC='uniform', T_HI=1.0,
    TAU_DENSITY=(0.0,), TAU_SPLIT=None, TAU_DENSITY_BC=True,
    PROFILE_SPACE='physical', REF_GRID='fv', REF_R_PTS=32, PDE_FORM='conservative',
    CONS_N_KSI=41, CONS_N_T=4, CONS_USE_IC=False, CONS_LEGACY=False,
    TRACK_PI_VAL=True, TIME_FEATURES='linear', ROLLOUT_K=0, LBFGS_PHASE=False,
    # defaults of the dataset and training sections; the paper's networks set their own in
    # configs/trained_model.json
    N_PDE=1000, NY_PDE=20000, N_IC=2000, NY_IC=100, N_BC=2000, NY_BC=3000, RESAMPLE_EVERY=50,
    EPOCHS=1000, PATIENCE=100, LR_DECAY=False, LR_DECAY_EPOCHS=None, LR_END_FACTOR=0.01, ES_FROM_EPOCH=0,
    MONITOR='pi_val_loss', MONITOR2=None,
)

ADAM_CODE = r'''
import json, logging, os, sys
sys.path.insert(0, %(root)r); sys.path.insert(0, %(scripts)r)
import train_nd as T
logging.basicConfig(level=logging.INFO, format="%%(message)s")
for k, v in %(recipe)s.items():                      # the fixed recipe
    setattr(T, k, v)
h = json.load(open(%(hpo)r))                          # architecture, lr and loss weights
assert int(h.get('n_sensors', T.N_SENSORS)) == T.N_SENSORS
T.HPO_CONFIG = os.path.relpath(%(hpo)r, %(root)r)
T.HIDDEN, T.N_LAYERS = int(h['hidden_dim']), int(h['number_of_hidden_layers'])
T.ACTIVATION, T.LR = h['activation'], float(h['lr'])
T.LOSS_WEIGHTS = [float(w) for w in h['loss_weights']]
T.NUM_BATCHES, T.NUM_BATCHES_HINT = %(num_batches)r, %(nb_floor)d   # 'auto': fewest that fit the free VRAM
T.HARD_BC = %(hard_bc)r
T.RUN_NAME, T.SEED = %(name)r, %(seed)d
T.RESUME = %(resume)r
for k, v in json.loads(%(overrides)r).items():        # the study's dataset, training and recipe keys
    setattr(T, k, tuple(v) if k == 'TAU_DENSITY' and v is not None else v)
logging.getLogger(__name__).info('Adam stage %%s | seed %%d | %%dx%%d %%s | weights %%s', T.RUN_NAME, T.SEED,
                                 T.HIDDEN, T.N_LAYERS, T.ACTIVATION, ['%%.4g' %% w for w in T.LOSS_WEIGHTS])
T.main()
'''


def run(cmd, dry, env=None):
    log.info('$ %s', ' '.join(c if len(c) < 90 else c[:87] + '...' for c in cmd))
    if not dry:
        subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def rel(p):
    return os.path.relpath(p, ROOT)


# ------------------------------------------------------------------ stages
def stage_hpo(cfg, study, dry):
    """Returns the folder whose model_config.json sets architecture, lr and loss weights."""
    h = cfg['hyperparameters']
    mode = h['mode']
    if mode == 'file':
        folder = os.path.join(ROOT, h['path'])
        if not os.path.exists(os.path.join(folder, 'model_config.json')):
            raise SystemExit('%s has no model_config.json' % h['path'])
        log.info('hyperparameters from %s', h['path'])
        return folder
    if mode == 'manual':
        folder = os.path.join(study, 'hyperparameters')
        os.makedirs(folder, exist_ok=True)
        mc = dict(model='MIONetND (v2, dimensionless)', n_sensors=21, hidden_dim=h['hidden_dim'],
                  number_of_hidden_layers=h['num_hidden_layers'], activation=h['activation'], lr=h['lr'],
                  loss_weights=[h['w_pde'], h['w_ic'], h['w_bc'], h['w_mc']], source='set by hand')
        json.dump(mc, open(os.path.join(folder, 'model_config.json'), 'w'), indent=2)
        log.info('hyperparameters set by hand: %s', mc)
        return folder
    if mode == 'run':
        folder = os.path.join(ROOT, 'runs', h['hpo_name'])
        if os.path.exists(os.path.join(folder, 'model_config.json')):
            log.info('HPO %s already done', h['hpo_name'])
            return folder
        if os.path.exists(folder):
            raise SystemExit('%s exists without model_config.json (interrupted HPO?) - rename it first' % folder)
        a = h.get('args', {})
        cmd = [PY, os.path.join('srcs', 'hpo_rollout.py'), '--name', h['hpo_name']]
        for k, v in a.items():
            flag = '--' + k.replace('_', '-')
            if isinstance(v, bool):
                cmd += [flag] if v else []
            elif isinstance(v, list):
                cmd += [flag] + [str(x) for x in v]
            else:
                cmd += [flag, str(v)]
        run(cmd, dry)
        return folder
    raise SystemExit("hyperparameters.mode must be 'file', 'manual' or 'run'")


def stage_adam(cfg, name, seed, hpo_folder, dry, resume=False):
    run_dir = os.path.join(RUNS, name)
    if os.path.exists(os.path.join(run_dir, 'model.pt')):
        log.info('%s exists - Adam stage skipped', name)
        return
    cont = os.path.exists(run_dir)
    if cont and not (resume and any(os.path.exists(os.path.join(run_dir, 'checkpoints', f))
                                    for f in ('last.pt', 'best.pt'))):
        raise SystemExit('%s exists without model.pt (interrupted?) - continue it with --resume, or rename '
                         'or delete it first' % name)
    ovr = {DATASET_KEYS[k]: v for k, v in cfg.get('dataset', {}).items() if k in DATASET_KEYS}
    t = cfg['training']
    ovr.update({ADAM_KEYS[k]: v for k, v in t.items() if k in ADAM_KEYS})
    nb = t.get('num_batches', 100)
    nb, nb_floor = ('auto', int(t.get('num_batches_min', 1))) if str(nb).lower() == 'auto' else (int(nb), int(nb))
    ovr.update(cfg.get('recipe', {}))                # advanced: any train_nd global, by its own name
    code = ADAM_CODE % dict(root=ROOT, scripts=SRCS, overrides=json.dumps(ovr), name=name, seed=seed,
                            recipe=repr(RECIPE),
                            hpo=os.path.join(hpo_folder, 'model_config.json'),
                            num_batches=nb, nb_floor=nb_floor, hard_bc=bool(t.get('hard_bc', False)),
                            resume=cont)
    log.info('Adam %s%s: %s', name, ' (resumed from its latest checkpoint)' if cont else '', ovr)
    run([PY, '-c', code], dry)


def stage_lbfgs(cfg, name, dry, resume=False):
    lb = cfg.get('lbfgs', {})
    if not lb.get('enabled', True):
        return
    if os.path.exists(os.path.join(RUNS, name + '_lbfgs', 'model.pt')):
        log.info('%s_lbfgs exists - L-BFGS stage skipped', name)
        return
    if resume and os.path.exists(os.path.join(RUNS, name + '_lbfgs')):   # interrupted: redo, keep the folder
        i, dst = 1, os.path.join(RUNS, name + '_lbfgs_interrupted')
        while os.path.exists(dst):
            i += 1
            dst = os.path.join(RUNS, '%s_lbfgs_interrupted%d' % (name, i))
        log.info('%s_lbfgs incomplete - renamed %s, refinement redone', name, os.path.basename(dst))
        if not dry:
            os.rename(os.path.join(RUNS, name + '_lbfgs'), dst)
    run([PY, os.path.join('srcs', 'refine_lbfgs.py'), '--from', name, '--init', 'model',
         '--monitor', lb.get('monitor', 'pi_val_loss'), '--epochs', str(lb.get('epochs', 16)),
         '--patience', str(lb.get('patience', 5)), '--pool', str(lb.get('pool', 6)), '--k', str(lb.get('k', 1)),
         '--c', str(lb.get('c', 1)), '--lr', str(lb.get('lr', 3e-4)), '--ny-pde', str(lb.get('ny_pde', 1000)),
         '--no-score'], dry)


def final_models(cfg, names):
    which = cfg.get('scoring', {}).get('models', 'both')
    lb = cfg.get('lbfgs', {}).get('enabled', True)
    out = []
    for n in names:
        if which in ('adam', 'both') or not lb:
            out.append(n)
        if lb and which in ('lbfgs', 'both'):
            out.append(n + '_lbfgs')
    return out


def scoring_tag(cfg):
    """The ds_mode tag the scoring writes, read back by the figures."""
    sc = cfg.get('scoring', {})
    return ('mean-adapt%g' % sc.get('tau_target', 0.05) + ('-direct' if sc.get('direct', True) else '')
            + ('-fv%d' % sc.get('fv_grid', 51) if sc.get('fv_grid', 51) else ''))


def stage_score(cfg, study, names, dry):
    sc = cfg.get('scoring', {})
    if not sc.get('enabled', True):
        return
    run([PY, os.path.join('srcs', 'evaluation.py'), '--models'] + final_models(cfg, names)
        + ['--tau-target', str(sc.get('tau_target', 0.05)), '--csv', os.path.join(study, 'results.csv')]
        + (['--direct'] if sc.get('direct', True) else [])
        + (['--fv-grid', str(sc.get('fv_grid', 51))] if sc.get('fv_grid', 51) else []), dry)


def stage_post(cfg, study, names, hpo_folder, dry):
    po = cfg.get('postprocess', {})
    figs = os.path.join(study, 'figures')
    os.makedirs(figs, exist_ok=True)
    env = dict(os.environ, MIONET_FIG_DIR=figs)
    lb = cfg.get('lbfgs', {}).get('enabled', True)
    if po.get('training_curves', True):
        for n in names:
            out = os.path.join(figs, n)
            os.makedirs(out, exist_ok=True)
            code = ('import os, sys; sys.path.insert(0, %r); import fig_training as F; '
                    'F.ADAM = %r; F.LBFGS = %r; F.FIG = %r; F.main()'
                    % (os.path.join(SCRIPTS, 'figures'), os.path.join(RUNS, n),
                       os.path.join(RUNS, n + ('_lbfgs' if lb else '')), out))
            run([PY, '-c', code], dry, env)
    if po.get('hpo', True) and os.path.exists(os.path.join(hpo_folder, 'study.db')):
        run([PY, os.path.join('scripts', 'figures', 'fig_hpo.py'), '--run', hpo_folder], dry, env)
    eb = po.get('error_bins', {})
    if eb.get('enabled', True) and cfg.get('scoring', {}).get('enabled', True):
        script = os.path.normpath(os.path.join(ROOT, eb.get('script', 'scripts/figures/fig_error_bins.py')))
        stage = eb.get('models', 'lbfgs' if lb else 'adam')
        models = [n + ('_lbfgs' if stage == 'lbfgs' else '') for n in names]
        cmd = [PY, script, '--by-temperature', '--metric', 'pct_full', '--csv', os.path.join(study, 'results.csv'),
               '--cv-models'] + models + ['--scoring', scoring_tag(cfg),
                                         '--no-box', '--suffix', '_' + cfg['name']]
        if eb.get('ylim'):
            cmd += ['--ylim', str(eb['ylim'])]
        run(cmd, dry, env)
        if po.get('phi_error', True):                # error against the peak flux, same seeds
            run([PY, os.path.join('scripts', 'figures', 'fig_phi_error.py'), '--csv', os.path.join(study, 'results.csv'),
                 '--models'] + models + ['--scoring', scoring_tag(cfg)],
                dry, env)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='study JSON (see configs/README.md)')
    ap.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--resume', action='store_true', help='continue interrupted runs instead of stopping')
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding='utf-8'))
    study = os.path.join(ROOT, 'runs', 'studies', cfg['name'])
    os.makedirs(study, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(study, 'study.log'))])
    cfg_copy = os.path.join(study, 'config.json')
    if os.path.exists(cfg_copy) and json.load(open(cfg_copy, encoding='utf-8')) != cfg:
        raise SystemExit('runs/studies/%s/config.json differs from %s: a study name is never reused with '
                         'another configuration - choose a new name' % (cfg['name'], a.config))
    json.dump(cfg, open(cfg_copy, 'w'), indent=2)
    if not os.path.exists(os.path.join(study, 'metadata.json')):
        json.dump(dict(date=datetime.datetime.now().isoformat(timespec='seconds'),
                       description=cfg.get('description', ''), dataset='physics-informed, generated (no labels)',
                       config=rel(a.config)), open(os.path.join(study, 'metadata.json'), 'w'), indent=2)
    names = ['%s_s%d' % (cfg['name'], s) for s in cfg.get('seeds', [42])]
    log.info('study %s | seeds %s | stages %s', cfg['name'], cfg.get('seeds', [42]), ' '.join(a.stages))

    hpo_folder = stage_hpo(cfg, study, a.dry_run) if 'hpo' in a.stages or cfg['hyperparameters']['mode'] != 'run' \
        else os.path.join(ROOT, 'runs', cfg['hyperparameters']['hpo_name'])
    for seed, n in zip(cfg.get('seeds', [42]), names):
        if 'adam' in a.stages:
            stage_adam(cfg, n, seed, hpo_folder, a.dry_run, a.resume)
        if 'lbfgs' in a.stages:
            stage_lbfgs(cfg, n, a.dry_run, a.resume)
        if 'score' in a.stages:
            stage_score(cfg, study, [n], a.dry_run)          # per seed: results appear as they come
    if 'post' in a.stages:
        stage_post(cfg, study, names, hpo_folder, a.dry_run)
    log.info('study %s done: %s', cfg['name'], rel(study))


if __name__ == '__main__':
    main()
