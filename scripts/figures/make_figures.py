"""
make_figures.py
===============
Regenerate the paper's data figures into figures/ (PDF), from the shipped runs and results:

    python scripts/figures/make_figures.py

Set MIONET_FIG_DIR to write them to another folder instead (e.g. the manuscript's figure folder).

  fig_hpo.pdf                     HPO history and fANOVA importance          runs/hpo_study/study.db
  fig_training.pdf                Adam + L-BFGS training history, seed 42    runs/v2_nondim/trained_model_s42{,_lbfgs}
  fig_training_data_nd.pdf        one realisation of the training samplers   runs/v2_nondim/trained_model_s42 (config)
  fig_fields_mohtat2020.pdf       one solution: Mohtat2020, 0 C, CC-CV 1C    PyBaMM (cached) + trained_model_s42_lbfgs
  fig_results_bins_byT_full1.pdf  full-profile roll-out error, four networks runs/studies/trained_model/results.csv
  fig_phi_error.pdf               the same error against the signed peak Phi runs/studies/trained_model/results.csv
  fig_cycles_dfn.pdf              the MIONet in SPM / SPMe / multi-slice vs a PyBaMM DFN, 1/2/3C cycles
  fig_surface_dfn.pdf             surface stoichiometry across the electrodes, DFN vs multi-slice, 3C
                                  (both: scripts/cell_models/cycles_vs_dfn.py, solves the DFN, ~5 min,
                                  and writes a new runs/cell_models/ai2020_NNN)

Scoring (srcs/evaluation.py, the score stage of scripts/run_training.py): 51-point radial grid, one
network call per window of tau <= 0.05 or a 5 % current change, the network evaluated directly at
the scoring radii, full-profile error without volume weighting (ds_mode
'mean-adapt0.05-direct-fv51'). The schematic figures of the paper are drawn separately.
"""
import logging
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = ['trained_model_s42_lbfgs', 'trained_model_s43_lbfgs', 'trained_model_s44_lbfgs', 'trained_model_s45_lbfgs']
JOBS = [
    ['fig_hpo.py'],
    ['fig_training.py'],
    ['fig_training_data.py'],
    ['fig_fields.py'],
    ['fig_error_bins.py', '--by-temperature', '--metric', 'pct_full', '--suffix', '_full1', '--cv-models', *SEEDS,
     '--no-box', '--scoring', 'mean-adapt0.05-direct-fv51'],
    ['fig_phi_error.py'],
    [os.path.join(os.pardir, 'cell_models', 'cycles_vs_dfn.py')],
]
log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    for job in JOBS:
        log.info('--- %s', job[0])
        subprocess.run([sys.executable, os.path.join(HERE, job[0]), *job[1:]], check=True)


if __name__ == '__main__':
    main()
