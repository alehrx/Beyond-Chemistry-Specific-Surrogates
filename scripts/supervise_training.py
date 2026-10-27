"""
supervise_training.py
=====================
Keeps a training study going through crashes - a GPU driver reset, for instance, kills every process
that uses CUDA, while this one does not (it never imports torch):

    python scripts/supervise_training.py configs/<study>.json
    python scripts/supervise_training.py configs/<study>.json --wait-pid 1234   # adopt a running study

It runs `scripts/run_training.py <config> --resume` until that exits cleanly. After a failure it waits
for the GPU to answer again (nvidia-smi), then relaunches: finished stages are skipped, an interrupted
Adam run continues from its latest checkpoint and an interrupted L-BFGS refinement is redone (see
run_training.py). It gives up after --max-restarts relaunches, or after three failures in a row with
no training progress (no new epoch, no new model), so a deterministic error does not loop forever.

With --wait-pid it first waits for an already running study process (e.g. one started before the
supervisor) and only takes over if that process ends without finishing the study.

Log: runs/studies/<name>/supervisor.log; the study's own output is appended to
runs/studies/<name>_console.log.
"""
import argparse
import ctypes
import json
import logging
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
RUNS = os.path.join(ROOT, 'runs', 'v2_nondim')
NVSMI = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32', 'nvidia-smi.exe')
log = logging.getLogger('supervise')


def wait_pid(pid):
    """Block until process `pid` exits (Windows: SYNCHRONIZE handle; elsewhere: poll)."""
    if os.name == 'nt':
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x00100000, False, int(pid))              # SYNCHRONIZE
        if not h:
            return                                                    # already gone
        k32.WaitForSingleObject(h, 0xFFFFFFFF)
        k32.CloseHandle(h)
        return
    while True:
        try:
            os.kill(int(pid), 0)
        except OSError:
            return
        time.sleep(30)


def study_done(study, name):
    p = os.path.join(study, 'study.log')
    if not os.path.exists(p):
        return False
    lines = [l for l in open(p, encoding='utf-8', errors='replace') if l.strip()]
    return bool(lines) and ('study %s done' % name) in lines[-1]


def gpu_ready(timeout_s=1800):
    """True once nvidia-smi answers (the driver is back after a reset); False after timeout_s."""
    if not os.path.exists(NVSMI):
        return True                                                   # nothing to check against
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if subprocess.run([NVSMI, '-L'], capture_output=True, timeout=60).returncode == 0:
                return True
        except (subprocess.SubprocessError, OSError):
            pass
        time.sleep(60)
    return False


def progress(cfg):
    """A number that grows while the study advances: epochs logged plus finished models."""
    n = 0
    for s in cfg.get('seeds', [42]):
        for d in ('%s_s%d' % (cfg['name'], s), '%s_s%d_lbfgs' % (cfg['name'], s)):
            m = os.path.join(RUNS, d, 'metrics.csv')
            if os.path.exists(m):
                n += sum(1 for _ in open(m, 'rb'))
            n += 1000000 * os.path.exists(os.path.join(RUNS, d, 'model.pt'))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='study JSON, as for run_training.py')
    ap.add_argument('--wait-pid', type=int, help='an already running study process to wait for first')
    ap.add_argument('--max-restarts', type=int, default=10)
    ap.add_argument('--pause', type=int, default=120, help='seconds to wait after a failure')
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding='utf-8'))
    study = os.path.join(ROOT, 'runs', 'studies', cfg['name'])
    os.makedirs(study, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(study, 'supervisor.log'))])
    log.info('supervising %s (pid %d)', a.config, os.getpid())
    if a.wait_pid:
        log.info('waiting for the running study process %d', a.wait_pid)
        wait_pid(a.wait_pid)
        if study_done(study, cfg['name']):
            log.info('the study finished - nothing to do')
            return
        log.info('process %d ended without finishing the study - taking over', a.wait_pid)
        time.sleep(a.pause)
    console = os.path.join(ROOT, 'runs', 'studies', cfg['name'] + '_console.log')
    restarts, stalled, last = 0, 0, progress(cfg)
    while True:
        if not gpu_ready():
            log.info('the GPU did not answer for 30 min - giving up'); sys.exit(1)
        log.info('launch %d: run_training.py %s --resume', restarts, a.config)
        with open(console, 'ab') as out:
            rc = subprocess.run([sys.executable, os.path.join('scripts', 'run_training.py'), a.config, '--resume'],
                                cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            log.info('study %s finished', cfg['name']); return
        now = progress(cfg)
        stalled = 0 if now > last else stalled + 1
        last = now
        restarts += 1
        log.info('run_training exited with code %d (restart %d of %d, %d failures without progress)',
                 rc, restarts, a.max_restarts, stalled)
        if restarts > a.max_restarts or stalled >= 3:
            log.info('giving up - see %s and the run folders', os.path.relpath(console, ROOT)); sys.exit(1)
        time.sleep(a.pause)


if __name__ == '__main__':
    main()
