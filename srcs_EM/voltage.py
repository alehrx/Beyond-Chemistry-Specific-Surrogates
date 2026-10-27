"""
voltage.py
==========
Terminal voltage on top of the surrogate's solid state - the measurement equation the
observer actually sees, and the only place where the electrolyte becomes observable.

    V = U_p(x_p^surf) - U_n(x_n^surf)          thermodynamics (+ entropic dU/dT correction)
        + eta_p - eta_n                        Butler-Volmer, eta = (2RT/F) asinh(i/(2 j0))
        + eta_conc                             electrolyte salt gradient      <- SPMe only
        + eta_ohm,e                            electrolyte ohmic drop
        - I R_contact

with I > 0 = discharge, so i_n = +I/(A a_n L_n) and i_p = -I/(A a_p L_p). This is PyBaMM's own
SPMe decomposition without its solid-phase ohmic drop, which is neglected: a series resistance
of ~0.03 mOhm for Ai2020, about 300 times smaller than the electrolyte's.

THE POINT OF THE `electrolyte` SWITCH
-------------------------------------
One object serves both models:

    CellVoltage.from_pybamm('Ai2020')                          -> SPM  + lumped R
    CellVoltage.from_pybamm('Ai2020', electrolyte=elyte)       -> SPMe

In SPM mode the electrolyte is frozen at c_e0 - constant exchange-current prefactor, no
concentration overpotential - and its ohmic drop is absorbed into a single series resistance
`r_lumped` (defaulting to the leading-order electrolyte resistance, i.e. the fairest possible
SPM baseline). In SPMe mode the same terms come from the live `Electrolyte` state: the
X-averaged c_e per electrode feeds j0, and eta_conc appears. The difference between the two
calls, at the SAME solid state, is the whole contribution of the electrolyte.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .chemistry import Chemistry, T_REF
from .electrolyte import Electrolyte

logger = logging.getLogger(__name__)


class CellVoltage:
    """Algebraic measurement equation for a cell, with or without an electrolyte state."""

    def __init__(self, param_set: str, chem_n: Chemistry, chem_p: Chemistry, params: dict,
                 electrolyte: Optional[Electrolyte] = None, r_lumped: Optional[float] = None,
                 r_contact: float = 0.0):
        self.param_set = param_set
        self.chem_n, self.chem_p = chem_n, chem_p
        self.p = params                                  # the PyBaMM ParameterValues object
        self.electrolyte = electrolyte
        self.A = chem_n.n_el * chem_n.A                  # total electrode area [m2]
        self.F = chem_n.F
        self.R_gas = float(params["Ideal gas constant [J.K-1.mol-1]"])
        self.c_e0 = float(params["Initial concentration in electrolyte [mol.m-3]"])
        self.r_contact = float(r_contact)
        self._r_lumped = r_lumped                        # SPM mode only; None -> leading order

    # ----------------------------------------------------------------- build
    @classmethod
    def from_pybamm(cls, param_set: str, electrolyte: Optional[Electrolyte] = None,
                    r_lumped: Optional[float] = None) -> "CellVoltage":
        """Build from a PyBaMM parameter set. Pass an `Electrolyte` for SPMe, omit it for SPM."""
        import pybamm
        p = pybamm.ParameterValues(param_set)
        chem_n = Chemistry.from_pybamm(param_set, "Negative")
        chem_p = Chemistry.from_pybamm(param_set, "Positive")
        try:
            r_contact = float(p["Contact resistance [Ohm]"])
        except KeyError:
            r_contact = 0.0
        return cls(param_set, chem_n, chem_p, p, electrolyte, r_lumped, r_contact)

    # --------------------------------------------------------------- helpers
    def _ev(self, fn, *args):
        """Evaluate a PyBaMM parameter function at floats, resolving any nested Parameters."""
        import pybamm
        val = fn(*[pybamm.Scalar(float(a)) for a in args]) if callable(fn) else fn
        return float(np.asarray(self.p.evaluate(val)).reshape(-1)[0])

    @property
    def r_lumped(self) -> float:
        """Series resistance used in SPM mode [Ohm] - the leading-order electrolyte ohmic
        resistance unless one was supplied explicitly."""
        if self._r_lumped is not None:
            return float(self._r_lumped)
        if self.electrolyte is not None:
            return self.electrolyte.r_ohmic()
        # leading-order electrolyte ohmic resistance from the parameter set alone
        e = Electrolyte.from_pybamm(self.param_set, n_modes=1, n_fv=32)
        self._r_lumped = e.r_ohmic()
        return self._r_lumped

    def ocp(self, electrode: str, sto: float, T: float = T_REF) -> float:
        """Open-circuit potential [V] including the entropic correction dU/dT (T - 298.15)."""
        el = electrode.capitalize()
        u = self._ev(self.p[f"{el} electrode OCP [V]"], sto)
        dudt_fn = self.p.get(f"{el} electrode OCP entropic change [V.K-1]", None)
        if dudt_fn is not None:
            u += self._ev(dudt_fn, sto) * (float(T) - 298.15)
        return u

    def j0(self, electrode: str, sto: float, c_e: float, T: float = T_REF) -> float:
        """Exchange-current density [A/m2]. c_e enters as j0 ~ c_e^0.5 - the second, quieter
        channel through which the electrolyte moves the voltage."""
        el = electrode.capitalize()
        chem = self.chem_n if el == "Negative" else self.chem_p
        fn = self.p[f"{el} electrode exchange-current density [A.m-2]"]
        return self._ev(fn, c_e, float(sto) * chem.c_max, chem.c_max, T)

    # --------------------------------------------------------------- voltage
    def terminal(self, x_n: float, x_p: float, I: float, T: float = T_REF,
                 q: Optional[np.ndarray] = None) -> dict:
        """Terminal voltage [V] and its decomposition.

        x_n, x_p : SURFACE stoichiometries of the two particles (the surrogate's x[-1])
        I        : cell current [A], > 0 on discharge
        q        : electrolyte modal state; required in SPMe mode, ignored in SPM mode
        """
        I = float(I)
        f = 2.0 * self.R_gas * float(T) / self.F

        if self.electrolyte is not None:
            if q is None:
                raise ValueError("SPMe mode needs the electrolyte state q (Electrolyte.step)")
            e = self.electrolyte.overpotential(q, I, T)
            c_e_n, c_e_p = e["c_e_n"], e["c_e_p"]
            eta_conc, eta_ohm_e = e["conc"], e["ohmic"]
        else:
            c_e_n = c_e_p = self.c_e0
            eta_conc, eta_ohm_e = 0.0, -I * self.r_lumped

        i_n = I / (self.A * self.chem_n.a * self.chem_n.L)        # [A/m2], > 0 on discharge
        i_p = -I / (self.A * self.chem_p.a * self.chem_p.L)
        eta_n = f * np.arcsinh(i_n / (2.0 * self.j0("Negative", x_n, c_e_n, T)))
        eta_p = f * np.arcsinh(i_p / (2.0 * self.j0("Positive", x_p, c_e_p, T)))

        u_n, u_p = self.ocp("Negative", x_n, T), self.ocp("Positive", x_p, T)
        v = u_p - u_n + eta_p - eta_n + eta_conc + eta_ohm_e - I * self.r_contact
        return {"V": float(v), "U_p": u_p, "U_n": u_n, "eta_p": float(eta_p), "eta_n": float(eta_n),
                "eta_conc": float(eta_conc), "eta_ohm_e": float(eta_ohm_e),
                "c_e_n": float(c_e_n), "c_e_p": float(c_e_p)}

    def V(self, x_n: float, x_p: float, I: float, T: float = T_REF,
          q: Optional[np.ndarray] = None) -> float:
        """Terminal voltage [V] only."""
        return self.terminal(x_n, x_p, I, T, q)["V"]

    def __repr__(self):
        mode = "SPMe" if self.electrolyte is not None else f"SPM + R={self.r_lumped * 1e3:.2f} mOhm"
        return f"CellVoltage({self.param_set}, {mode})"
