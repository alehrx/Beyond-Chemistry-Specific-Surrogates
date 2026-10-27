"""
cell.py
=======
The whole cell as one recursive state: two surrogate particles + the electrolyte + the
measurement equation. This is what an observer actually propagates.

    state = (x_n, x_p, q)     x_n, x_p : (n_sensors,) radial stoichiometry profiles
                              q        : (n_modes,)   electrolyte modal coordinates

    x_n, x_p  <- Surrogate.step   (the trained dimensionless operator, one net, both electrodes)
    q         <- Electrolyte.step (exponential integrator, exact at constant current)
    V         <- CellVoltage.terminal(x_n[-1], x_p[-1], I, T, q)

Every sub-step is driven by the SAME (I, dt, T) and nothing is solved implicitly: under the
single-particle assumption the reaction current is uniform and known from I, so the particles
and the electrolyte are three independent driven systems that only meet in the algebraic
voltage. That is what makes the electrolyte two extra states rather than a solver.

SIGN CONVENTION: I > 0 is DISCHARGE throughout (as in `Chemistry.flux` and `CellVoltage`).
Laboratory files usually store the opposite - negate on the way in.

SPM vs SPMe is one flag:

    Cell.from_run(run, 'Mohtat2020', electrolyte=True)    # SPMe: q is live, eta_conc appears
    Cell.from_run(run, 'Mohtat2020', electrolyte=False)   # SPM : q is empty, lumped R instead

so the two models share the identical particle rollout and differ only in the electrolyte.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np

from .chemistry import Chemistry, T_REF
from .electrolyte import Electrolyte
from .surrogate import Surrogate
from .voltage import CellVoltage

logger = logging.getLogger(__name__)


@dataclass
class CellState:
    """One recursive state: both radial profiles and the electrolyte modal coordinates.

    `hold_dt` / `hold_Idt` buffer the charge waiting to be pushed into the particles when the
    cell runs multi-rate (see `Cell.dt_particle`); they are zero in single-rate operation.
    """
    x_n: np.ndarray
    x_p: np.ndarray
    q: np.ndarray
    hold_dt: float = 0.0
    hold_Idt: float = 0.0

    def copy(self) -> "CellState":
        return CellState(self.x_n.copy(), self.x_p.copy(), self.q.copy(),
                         self.hold_dt, self.hold_Idt)

    @property
    def surfaces(self):
        """(x_n, x_p) at the particle surface - what the voltage equation sees."""
        return float(self.x_n[-1]), float(self.x_p[-1])


class Cell:
    """Surrogate particles + optional electrolyte + terminal voltage, stepped together."""

    def __init__(self, surrogate: Surrogate, voltage: CellVoltage,
                 electrolyte: Optional[Electrolyte] = None, ds_mode: str = 'ref'):
        self.sur = surrogate
        self.volt = voltage
        self.elyte = electrolyte
        self.ds_mode = ds_mode
        self.chem_n: Chemistry = voltage.chem_n
        self.chem_p: Chemistry = voltage.chem_p
        self.param_set = voltage.param_set

    # ------------------------------------------------------------------ build
    @classmethod
    def from_run(cls, run_dir: str, param_set: str, electrolyte: bool = True,
                 n_modes: int = 2, tdf_mode: str = 'quad', r_lumped: Optional[float] = None,
                 ds_mode: str = 'ref', **elyte_kw) -> "Cell":
        """Load the trained surrogate and assemble a cell for `param_set`.

        electrolyte=False gives the SPM baseline: the same particles, no electrolyte state, and
        a single series resistance (`r_lumped`, defaulting to the leading-order electrolyte
        ohmic resistance - the fairest SPM can be made without fitting).
        """
        sur = Surrogate.from_run(run_dir)
        ely = Electrolyte.from_pybamm(param_set, n_modes=n_modes, tdf_mode=tdf_mode,
                                      **elyte_kw) if electrolyte else None
        volt = CellVoltage.from_pybamm(param_set, electrolyte=ely, r_lumped=r_lumped)
        return cls(sur, volt, ely, ds_mode=ds_mode)

    # ------------------------------------------------------------------ state
    def initial_state(self, soc) -> CellState:
        """Equilibrated state at `soc` - a fraction in [0, 1] or a string like '4.04 V'.

        Both particles are uniform (a rested cell has no radial gradient) and the electrolyte
        is at c_e0 (q = 0), which is its own rest state.
        """
        import pybamm
        x_n, x_p = pybamm.lithium_ion.get_initial_stoichiometries(soc, self.volt.p)
        x_n = float(np.asarray(x_n).reshape(-1)[0])
        x_p = float(np.asarray(x_p).reshape(-1)[0])
        n = self.sur.n
        q = np.zeros(self.elyte.n_modes) if self.elyte is not None else np.zeros(0)
        return CellState(np.full(n, x_n), np.full(n, x_p), q)

    def soc(self, st: CellState) -> float:
        """Volume-mean negative-electrode stoichiometry - the coulomb-counting state."""
        return self.sur.volume_mean(st.x_n)

    # ------------------------------------------------------------------- step
    def step(self, st: CellState, I: float, dt: float, T: float = T_REF) -> CellState:
        """Advance the whole state by dt [s] at constant current I [A] (I > 0 = discharge)."""
        if dt <= 0:
            return st
        x_n = self.sur.step(st.x_n, I, dt, self.chem_n, T, ds_mode=self.ds_mode)
        x_p = self.sur.step(st.x_p, I, dt, self.chem_p, T, ds_mode=self.ds_mode)
        q = self.elyte.step(st.q, I, dt, T) if self.elyte is not None else st.q
        return CellState(x_n, x_p, q)

    def terminal(self, st: CellState, I: float, T: float = T_REF) -> dict:
        """Terminal voltage and its decomposition at the current state."""
        x_n, x_p = st.surfaces
        return self.volt.terminal(x_n, x_p, I, T, st.q if self.elyte is not None else None)

    # ---------------------------------------------------------------- rollout
    def rollout(self, t: Sequence[float], I: Sequence[float], T=T_REF,
                state0: Optional[CellState] = None, soc0=None, keep_states: bool = False) -> dict:
        """Run the cell over a measured (t, I, T) record.

        The current is held constant over each interval and applied as I[k] on (t[k-1], t[k]],
        which is what a zero-order-hold logger records. Returns the voltage trace, its
        decomposition, the surface stoichiometries and the electrolyte concentrations.
        """
        t = np.asarray(t, dtype=float)
        I = np.asarray(I, dtype=float)
        T = np.full(len(t), float(T)) if np.isscalar(T) else np.asarray(T, dtype=float)
        if state0 is None:
            if soc0 is None:
                raise ValueError('rollout needs state0 or soc0')
            state0 = self.initial_state(soc0)
        st = state0.copy()

        n = len(t)
        keys = ('V', 'U_p', 'U_n', 'eta_p', 'eta_n', 'eta_conc', 'eta_ohm_e', 'c_e_n', 'c_e_p')
        out = {k: np.zeros(n) for k in keys}
        out['x_n_surf'] = np.zeros(n)
        out['x_p_surf'] = np.zeros(n)
        out['soc_n'] = np.zeros(n)
        states = []
        for k in range(n):
            if k > 0:
                st = self.step(st, float(I[k]), float(t[k] - t[k - 1]), float(T[k]))
            d = self.terminal(st, float(I[k]), float(T[k]))
            for key in keys:
                out[key][k] = d[key]
            out['x_n_surf'][k], out['x_p_surf'][k] = st.surfaces
            out['soc_n'][k] = self.soc(st)
            if keep_states:
                states.append(st.copy())
        out['t'], out['I'] = t, I
        out['state'] = st
        if keep_states:
            out['states'] = states
        return out

    # --------------------------------------------------------------- fitting
    @staticmethod
    def fit_series_resistance(V_model: np.ndarray, V_meas: np.ndarray, I: np.ndarray) -> float:
        """The single series resistance that best explains the residual [Ohm].

        V = V_model - I R is linear in R, so this is a one-line least squares. Real cells
        carry tab, current-collector and contact resistance that no P2D parameter set
        includes; giving EVERY model its own fitted R is what makes an SPM-vs-SPMe comparison
        on real data a fair one - the SPM baseline gets the best resistance it could
        possibly have.
        """
        I = np.asarray(I, dtype=float)
        denom = float(np.sum(I ** 2))
        if denom <= 0:
            return 0.0
        return float(np.sum(I * (np.asarray(V_model) - np.asarray(V_meas))) / denom)

    def __repr__(self):
        mode = 'SPMe' if self.elyte is not None else 'SPM'
        return (f"Cell({self.param_set}, {mode}, n_sensors={self.sur.n}, "
                f"n_modes={0 if self.elyte is None else self.elyte.n_modes})")
