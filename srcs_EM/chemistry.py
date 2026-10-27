"""
chemistry.py
============
The transform layer between a physical electrode and the DIMENSIONLESS single-
particle diffusion problem the v2 surrogate is trained on:

    dx/dtau = xi^-2 d/dxi ( xi^2 dx/dxi ),        -dx/dxi |_{xi=1} = Phi(tau)

    xi  = r / R                         tau = D_s(T) * t / R^2        x = c / c_max
    Phi = sign * I / j0_aux(T)          j0_aux = n_el * A * D_s(T) * c_max * F * a * L / R
    a   = 3 * eps / R

so that Phi = j * R / (D_s * c_max), with j = I / (n_el * A * a * L * F) the surface
molar flux. EVERY chemistry parameter (R, eps, L, A, n_el, c_max) and TEMPERATURE
(through D_s(T)) enter only here. The network never sees seconds, amperes, or a
parameter set - which is what makes one trained surrogate serve every constant-D_s
chemistry at every temperature.

Backward equivalence: at 298.15 K, `flux()` equals the legacy hard-BC flux
i*C/j0_aux of srcs/utils.compute_bc2_constants (sign included), so a v1 network sees
identical inputs at the training reference.

D_s and stoichiometry: `ds_x_ratio` (max/min of D_s over x at T_REF) is recorded at
construction. Every transform accepts an optional `sto`: the PyBaMM diffusivity function is
then evaluated at (sto, T) - at deployment the surrogate can pass the current profile's mean or
surface stoichiometry EVERY STEP (Surrogate.step ds_mode). For constant-D_s chemistries
(Mohtat2020, Ai2020, Marquis2019: ratio 1.00) this is identical to the reference value; for a
D_s(x) chemistry it is a per-step quasi-linearisation (the PDE inside the step still uses one
D_s), exact in the limit of slowly varying D_s.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)
T_REF = 298.15                       # training reference temperature [K]


@dataclass
class Chemistry:
    """Everything needed to map (I [A], dt [s], T [K]) <-> (Phi, tau) for one electrode."""
    electrode: str                   # 'Negative' | 'Positive'
    param_set: str                   # provenance, e.g. 'Mohtat2020'
    R: float                         # particle radius [m]
    eps: float                       # active-material volume fraction [-]
    L: float                         # electrode thickness [m]
    A: float                         # electrode area W*H [m2]
    n_el: float                      # electrodes in parallel [-]
    c_max: float                     # max solid concentration [mol/m3]
    F: float                         # Faraday [C/mol]
    sign: float                      # +1 Negative, -1 Positive (legacy convention)
    ds_ref: float                    # D_s at T_REF [m2/s]
    ds_x_ratio: float = 1.0          # max/min of D_s over stoichiometry at T_REF
    sto_ref: float = 0.5             # stoichiometry used when `sto` is not given
    _ds_func: Optional[Callable[[float, float], float]] = field(default=None, repr=False, compare=False)  # (sto, T)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_pybamm(cls, param_set: str, electrode: str, sto_ref: float = 0.5) -> "Chemistry":
        """Build from a PyBaMM parameter set (same keys as utils.compute_bc2_constants)."""
        import pybamm
        p = pybamm.ParameterValues(param_set)
        el = electrode.capitalize()
        # PyBaMM renamed "<el> electrode diffusivity" -> "<el> particle diffusivity"; the old name
        # still resolves through the deprecation alias, but newer sets (Chen2020, Prada2013) store a
        # plain CONSTANT there rather than a callable of (sto, T) - no stoichiometry and, more
        # importantly, no Arrhenius temperature dependence. Accept all three shapes.
        Diff = None
        for key in (f"{el} particle diffusivity [m2.s-1]", f"{el} electrode diffusivity [m2.s-1]"):
            try:
                Diff = p[key]
                break
            except KeyError:
                continue
        if Diff is None:
            raise KeyError(f"{param_set}: no solid diffusivity for the {el} electrode")

        if callable(Diff):
            def ds_func(sto: float, T: float) -> float:
                v = Diff(float(sto), float(T))
                return float(v.evaluate() if hasattr(v, "evaluate") else v)
        else:
            d_const = float(Diff.value if hasattr(Diff, "value") else Diff)

            def ds_func(sto: float, T: float) -> float:
                return d_const
            logger.warning("%s %s: D_s is a constant %.3e m2/s in this parameter set - it carries no "
                           "stoichiometry and no Arrhenius temperature dependence, so tau and Phi do "
                           "NOT move with T for this electrode.", param_set, el, d_const)

        # 7 points missed the sharp features of a graphite D_s: it reported 38x for the
        # ORegan2022 anode where a 193-point sweep finds 380x. The curve is cheap to evaluate and
        # this number is quoted in the warning and in the results CSV, so it should not be a
        # coarse-sampling artefact.
        stos = np.linspace(0.02, 0.98, 193)
        d_over_x = [ds_func(float(s), T_REF) for s in stos]
        ratio = max(d_over_x) / min(d_over_x)
        if ratio > 1.05:
            logger.warning("%s %s: D_s varies %.2fx over stoichiometry - the dimensionless "
                           "collapse is only approximate (evaluated at sto=%.2f).",
                           param_set, el, ratio, sto_ref)
        return cls(
            electrode=el, param_set=param_set,
            R=float(p[f"{el} particle radius [m]"]),
            eps=float(p[f"{el} electrode active material volume fraction"]),
            L=float(p[f"{el} electrode thickness [m]"]),
            A=float(p["Electrode width [m]"]) * float(p["Electrode height [m]"]),
            n_el=float(p["Number of electrodes connected in parallel to make a cell"]),
            c_max=float(p[f"Maximum concentration in {el.lower()} electrode [mol.m-3]"]),
            F=float(p["Faraday constant [C.mol-1]"]),
            sign=1.0 if el == "Negative" else -1.0,
            ds_ref=ds_func(sto_ref, T_REF), ds_x_ratio=ratio, sto_ref=float(sto_ref), _ds_func=ds_func,
        )

    # ------------------------------------------------------------- properties
    @property
    def a(self) -> float:
        """Specific interfacial area 3*eps/R [1/m]."""
        return 3.0 * self.eps / self.R

    # Tabulated D_s. PyBaMM's diffusivity is a SYMBOLIC expression and evaluating it costs ~45 us,
    # which `to_dimless` pays TWICE per step (flux + fourier) - 90 us, larger than either the
    # network forward or the finite-volume solve it is wrapped around. `enable_ds_cache` replaces
    # it with a table over stoichiometry, built once per temperature.
    #   ds_x_ratio == 1 (Mohtat, Ai, Marquis, Chen, Prada): D_s is constant in sto, so the table
    #   is flat and the substitution is EXACT.
    #   ds_x_ratio > 1 (ORegan): linear interpolation on `n_sto` points, error reported by
    #   `ds_cache_error()`. Opt-in for exactly this reason - it must not silently perturb a
    #   published number.
    _DS_CACHE_MAX_T = 64

    def enable_ds_cache(self, n_sto: int = 2048) -> "Chemistry":
        """Tabulate D_s over stoichiometry, per temperature, built lazily on first use."""
        object.__setattr__(self, "_ds_tab", {})
        object.__setattr__(self, "_ds_n", int(n_sto))
        return self

    def _ds_table(self, T: float):
        tab = self._ds_tab.get(T)
        if tab is None:
            if len(self._ds_tab) >= self._DS_CACHE_MAX_T:
                self._ds_tab.clear()                      # bounded: never grow without limit
            g = np.linspace(0.0, 1.0, self._ds_n)
            tab = self._ds_tab[T] = (g, np.array([float(self._ds_func(float(s), T)) for s in g]))
        return tab

    def ds_cache_error(self, T: float = T_REF, n_probe: int = 5000) -> float:
        """Max relative error the table introduces at T. Zero for a constant-D_s parameter set."""
        if getattr(self, "_ds_tab", None) is None or self._ds_func is None:
            return 0.0
        g, v = self._ds_table(T)
        s = np.linspace(0.0, 1.0, n_probe)
        exact = np.array([float(self._ds_func(float(x), T)) for x in s])
        return float(np.max(np.abs(np.interp(s, g, v) - exact) / np.abs(exact)))

    def Ds(self, T: float = T_REF, sto: Optional[float] = None) -> float:
        """Solid diffusivity [m2/s] at T and stoichiometry `sto` (default sto_ref) - the PyBaMM
        function evaluated directly. Falls back to ds_ref without a callable (e.g. from a manifest)."""
        if self._ds_func is None:
            return self.ds_ref
        s = self.sto_ref if sto is None else float(np.clip(sto, 0.0, 1.0))
        if getattr(self, "_ds_tab", None) is not None:
            g, v = self._ds_table(T)
            return float(np.interp(s, g, v))
        return float(self._ds_func(s, T))

    def j0_aux(self, T: float = T_REF, sto: Optional[float] = None) -> float:
        """Flux normaliser n_el*A*D_s(T,sto)*c_max*F*a*L/R [A] - the denominator of Phi.
        At T_REF (and sto_ref) this is exactly utils.compute_bc2_constants' j0_aux."""
        return self.n_el * self.A * self.Ds(T, sto) * self.c_max * self.F * self.a * self.L / self.R

    # ------------------------------------------------------------- transforms
    def flux(self, I: float, T: float = T_REF, sto: Optional[float] = None) -> float:
        """Dimensionless surface flux Phi = sign*I/j0_aux(T, sto)  (+ = discharge convention)."""
        return self.sign * float(I) / self.j0_aux(T, sto)

    def fourier(self, dt: float, T: float = T_REF, sto: Optional[float] = None) -> float:
        """Dimensionless time tau = D_s(T, sto)*dt/R^2."""
        return self.Ds(T, sto) * float(dt) / self.R ** 2

    def to_dimless(self, I: float, dt: float, T: float = T_REF, sto: Optional[float] = None):
        """(I [A], dt [s], T [K][, stoichiometry]) -> (Phi, tau)."""
        return self.flux(I, T, sto), self.fourier(dt, T, sto)

    def from_dimless(self, Phi: float, tau: float, T: float = T_REF):
        """(Phi, tau, T [K]) -> (I [A], dt [s]). Inverse of to_dimless."""
        return self.sign * float(Phi) * self.j0_aux(T), float(tau) * self.R ** 2 / self.Ds(T)

    # -------------------------------------------------------------- manifest
    def to_dict(self) -> dict:
        """Manifest-safe (no callable). Rebuild the callable with from_pybamm when needed."""
        d = asdict(self)
        d.pop("_ds_func", None)
        d["a"] = self.a
        d["j0_aux_ref"] = self.j0_aux(T_REF)
        return d
