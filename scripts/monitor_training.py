"""
monitor_training.py
===================
Live monitoring of a training study in TensorBoard (a web page in the browser). Set the variables
below and run, in a second terminal, while the study trains:

    python scripts/monitor_training.py

Every REFRESH_S seconds the new rows of each run's metrics.csv (runs/v2_nondim/<study>_s<seed>[_lbfgs])
are copied into TensorBoard logs under runs/studies/<study>/tensorboard/, so the training itself
is not touched and a study that is already running can be watched. The page shows, per seed:
the training and held-out physics losses, the four loss terms and the lithium balance; the L-BFGS stage continues the Adam curves on the same epoch axis.
Stop with Ctrl+C (the training keeps running).
"""
import glob
import logging
import os
import subprocess
import sys
import time
import webbrowser

import pandas as pd

# ============================== what to monitor ==============================
STUDY = 'trained_model'  # the "name" of the study's JSON
REFRESH_S = 30           # how often the logs are updated [s]
PORT = 6006              # the page is http://localhost:<PORT>
OPEN_BROWSER = True
# ==============================================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
RUNS = os.path.join(ROOT, 'runs', 'v2_nondim')
TB = os.path.join(ROOT, 'runs', 'studies', STUDY, 'tensorboard')
TAGS = {'train_loss': 'loss/training', 'pi_val_loss': 'loss/validation (PI, held-out)',
        'pde_loss': 'terms/PDE', 'ic_loss': 'terms/IC', 'bc_loss': 'terms/BC', 'cons_loss': 'terms/MC',
        'cons_val': 'diagnostics/lithium balance (RMS)'}
log = logging.getLogger('monitor')


def read(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None


def main():
    from torch.utils.tensorboard import SummaryWriter
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
    os.makedirs(TB, exist_ok=True)
    board = subprocess.Popen([sys.executable, '-m', 'tensorboard.main', '--logdir', TB, '--port', str(PORT),
                              '--reload_interval', str(REFRESH_S)], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    url = 'http://localhost:%d' % PORT
    log.info('TensorBoard for study %s at %s (Ctrl+C to stop the monitor; the training keeps running)', STUDY, url)
    if OPEN_BROWSER:
        time.sleep(5)
        webbrowser.open(url)
    writers, done = {}, {}
    try:
        while True:
            status = []
            for adam in sorted(d for d in glob.glob(os.path.join(RUNS, STUDY + '_s*')) if not d.endswith('_lbfgs')):
                seed = os.path.basename(adam)[len(STUDY) + 2:]
                a = read(os.path.join(adam, 'metrics.csv'))
                b = read(os.path.join(adam + '_lbfgs', 'metrics.csv'))
                n_a = 0 if a is None else len(a)
                for stage, df, offset in (('adam', a, 0), ('lbfgs', b, n_a)):
                    if df is None or df.empty:
                        continue
                    key = (seed, stage)
                    w = writers.get(key) or writers.setdefault(key, SummaryWriter(os.path.join(TB, 'seed%s_%s' % (seed, stage))))
                    for _, row in df.iloc[done.get(key, 0):].iterrows():
                        for col, tag in TAGS.items():
                            if col in row and pd.notna(row[col]):
                                w.add_scalar(tag, float(row[col]), int(row['epoch']) + offset)
                    w.flush()
                    done[key] = len(df)
                if a is not None and not a.empty:
                    i = a.pi_val_loss.idxmin()
                    s = 'seed %s: Adam epoch %d, best PI val %.3g (epoch %d)' % (seed, a.epoch.iloc[-1], a.pi_val_loss[i], a.epoch[i])
                    if b is not None and not b.empty:
                        s += ' | L-BFGS epoch %d, best %.3g' % (b.epoch.iloc[-1], b.pi_val_loss.min())
                    status.append(s)
            log.info(' || '.join(status) or 'no run of %s yet' % STUDY)
            time.sleep(REFRESH_S)
    except KeyboardInterrupt:
        pass
    finally:
        for w in writers.values():
            w.close()
        board.terminate()


if __name__ == '__main__':
    main()
