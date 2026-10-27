"""
electrolyte.py
==============
The term the single-particle model cannot represent: the salt gradient in the liquid.

WHY IT EXISTS
-------------
`chemistry.py` + `surrogate.py` solve the SOLID phase - lithium inside the particles. The
liquid only appears there as an assumption: that it is uniform and infinitely conductive.
It is not. Only the fraction t_+ of the ionic current is carried by Li+ migration; the anion
carries the rest but cannot leave the cell, so it redistributes until its own gradient stalls
it. Electroneutrality drags the salt with it, and a concentration profile builds across the
cell on the timescale L^2/D_e - which for a typical cell is the SAME order as the particle
timescale R^2/D_s. That is why it cannot be lumped into a series resistance: it lives in the
frequency band the observer uses to separate SOC from capacity.

THE MODEL (SPMe electrolyte, leading order)
-------------------------------------------
    eps_k dc_e/dt = d/dx ( D_e eps_k^b_k dc_e/dx ) + (1 - t_+)/F di_e/dx ,

on the three-domain slab [0, L_n + L_s + L_p], zero flux at both current collectors, c_e and
flux continuous at the interfaces. Under the single-particle assumption the reaction current
is uniform in each electrode, so di_e/dx is a KNOWN piecewise-constant function of the cell
current I: +I/(A L_n) in the negative, 0 in the separator, -I/(A L_p) in the positive
(I > 0 = discharge, so the negative side gets richer). The coupling is one-way - c_e never
feeds back on the reaction distribution - which is exactly what keeps this cheap.

HOW IT IS SOLVED
----------------
Finite volumes -> M c' = K c + B I, with M = diag(eps_i dx_i), K the symmetric
negative-semidefinite diffusion matrix (harmonic face averaging, so the porosity jumps at the
interfaces are resolved as a series resistance) and B the source. Then the generalised
symmetric eigenproblem K phi = lambda M phi, M-orthonormal, truncated to `n_modes`:

    c_e(x, t) = c_e0 + sum_i q_i(t) phi_i(x) ,      q_i' = D_e(T) lam_i q_i + g_i I

with g_i = phi_i^T B. Constant I over a step integrates EXACTLY (exponential integrator), so
a step is unconditionally stable at any dt and there is no sub-stepping.

TWO STRUCTURAL FACTS, both used here:
  * TEMPERATURE FACTORS OUT. K = D_e(T) * K_hat because D_e enters every face the same way,
    so the eigendecomposition is computed ONCE and T only rescales the eigenvalues - the same
    collapse `chemistry.py` uses for D_s(T), and the reason one decomposition serves all
    temperatures.
  * SALT IS EXACTLY CONSERVED. sum_i B_i = 0 (what leaves the negative enters the positive),
    so the lambda = 0 eigenvector - the constant mode - is never excited and the cell-wide
    salt inventory is invariant by construction. This is the electrolyte's analogue of the
    sphere's mass correction, and it needs no correction at all: it is exact in the modal
    basis (checked at build time, and by `salt_error`).

WHAT IS LINEARISED
------------------
D_e, kappa and the thermodynamic factor are evaluated at (c_e0, T) - the leading-order SPMe
assumption. The concentration overpotential can optionally be integrated with the LOCAL
thermodynamic factor (`tdf_mode='quad'`), which removes most of the residual against PyBaMM.

Ai2020 WARNING
--------------
PyBaMM 25.8 ships `Ai2020`'s electrolyte diffusivity as the LIONSIMBA/Capiglia expression
WITHOUT the 1e-4 cm^2/s -> m^2/s conversion that the same source applies to the conductivity:
D_e(1 M, 25 C) comes out 3.2e-6 m2/s instead of 3.2e-10. At that value the electrolyte is
uniform to machine precision and PyBaMM's own SPMe silently degenerates to SPM.
`de_scale='auto'` restores the factor for this parameter set (loudly); pass an explicit float
to override.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

T_REF = 298.15                        # reference temperature [K] (shared with chemistry.py)
T_TABLE = (223.15, 353.15, 1.0)       # (min, max, step) [K] of the transport-property table

# PyBaMM parameter sets whose electrolyte diffusivity is published in cm2/s under an
# [m2.s-1] label. Applied by de_scale='auto'; see the module docstring.
DE_UNIT_FIX = {"Ai2020": 1e-4}


@dataclass
class Electrolyte:
    """Three-domain SPMe electrolyte as a small modal state space, built from a PyBaMM set."""
    param_set: str
    L_n: float                        # negative electrode thickness [m]
    L_s: float                        # separator thickness [m]
    L_p: float                        # positive electrode thickness [m]
    eps_n: float                      # porosity (liquid volume fraction) [-]
    eps_s: float
    eps_p: float
    b_n: float                        # Bruggeman exponent (electrolyte) [-]
    b_s: float
    b_p: float
    A: float                          # TOTAL electrode area n_el*W*H [m2]
    c_e0: float                       # initial / reference salt concentration [mol.m-3]
    t_plus: float                     # cation transference number [-]
    F: float                          # Faraday [C/mol]
    R_gas: float                      # ideal gas constant [J/(mol K)]
    de_scale: float = 1.0             # unit correction applied to the PyBaMM D_e function
    n_modes: int = 4                  # retained eigenmodes
    n_fv: int = 150                   # finite-volume cells used to build them
    tdf_mode: str = "ref"             # 'ref' (TDF at c_e0) | 'quad' (local TDF, log-c quadrature)
    de_ref: float = 0.0               # D_e(c_e0, T_REF) [m2/s], after de_scale
    kappa_ref: float = 0.0            # kappa(c_e0, T_REF) [S/m]
    tdf_ref: float = 1.0              # 1 + dln f / dln c at (c_e0, T_REF)
    _de_func: Optional[Callable[[float, float], float]] = field(default=None, repr=False, compare=False)
    _kappa_func: Optional[Callable[[float, float], float]] = field(default=None, repr=False, compare=False)
    _tdf_func: Optional[Callable[[float, float], float]] = field(default=None, repr=False, compare=False)
    # Runtime multiplier on D_e, for IDENTIFICATION. Distinct from de_scale, which is a
    # fixed unit correction applied once when the parameter functions are built: this one is
    # a free parameter a fit may move, and every derived quantity follows it because they
    # all read De() (tau_e included). Scaling here rather than rebuilding matters because
    # the eigendecomposition does not depend on D_e at all - K = D_e * K_hat - so a new
    # value costs nothing and needs no rebuild.
    de_mult: float = 1.0

    # --------------------------------------------------------------- build
    @classmethod
    def from_pybamm(cls, param_set: str, n_modes: int = 4, n_fv: int = 150,
                    de_scale="auto", tdf_mode: str = "ref") -> "Electrolyte":
        """Build from a PyBaMM parameter set. `de_scale`: 'auto' applies DE_UNIT_FIX, or a float."""
        import pybamm
        p = pybamm.ParameterValues(param_set)

        if de_scale == "auto":
            de_scale = DE_UNIT_FIX.get(param_set, 1.0)
            if de_scale != 1.0:
                logger.warning("%s: applying de_scale=%.0e to the PyBaMM electrolyte diffusivity - "
                               "the shipped function is in cm2/s under an [m2.s-1] label. Pass "
                               "de_scale=1.0 to reproduce PyBaMM as-is (electrolyte then inert).",
                               param_set, de_scale)
        de_scale = float(de_scale)

        def _ev(fn, *args):
            import pybamm as _pb
            val = fn(*[_pb.Scalar(float(a)) for a in args]) if callable(fn) else fn
            return float(np.asarray(p.evaluate(val)).reshape(-1)[0])

        de_fn = p["Electrolyte diffusivity [m2.s-1]"]
        ka_fn = p["Electrolyte conductivity [S.m-1]"]
        td_fn = p.get("Thermodynamic factor", None)

        def de_func(c, T):
            return de_scale * _ev(de_fn, c, T)

        def kappa_func(c, T):
            return _ev(ka_fn, c, T)

        def tdf_func(c, T):
            return 1.0 if td_fn is None else _ev(td_fn, c, T)

        c_e0 = float(p["Initial concentration in electrolyte [mol.m-3]"])
        obj = cls(
            param_set=param_set,
            L_n=float(p["Negative electrode thickness [m]"]),
            L_s=float(p["Separator thickness [m]"]),
            L_p=float(p["Positive electrode thickness [m]"]),
            eps_n=float(p["Negative electrode porosity"]),
            eps_s=float(p["Separator porosity"]),
            eps_p=float(p["Positive electrode porosity"]),
            b_n=float(p["Negative electrode Bruggeman coefficient (electrolyte)"]),
            b_s=float(p["Separator Bruggeman coefficient (electrolyte)"]),
            b_p=float(p["Positive electrode Bruggeman coefficient (electrolyte)"]),
            A=float(p["Electrode width [m]"]) * float(p["Electrode height [m]"])
              * float(p["Number of electrodes connected in parallel to make a cell"]),
            c_e0=c_e0,
            t_plus=_ev(p["Cation transference number"]),
            F=float(p["Faraday constant [C.mol-1]"]),
            R_gas=float(p["Ideal gas constant [J.K-1.mol-1]"]),
            de_scale=de_scale, n_modes=int(n_modes), n_fv=int(n_fv), tdf_mode=str(tdf_mode),
            _de_func=de_func, _kappa_func=kappa_func, _tdf_func=tdf_func,
        )
        obj.de_ref = de_func(c_e0, T_REF)
        obj.kappa_ref = kappa_func(c_e0, T_REF)
        obj.tdf_ref = tdf_func(c_e0, T_REF)
        obj._build_modes()
        return obj

    # --------------------------------------------------------- discretisation
    def _grid(self):
        """Cell widths, porosity, transport efficiency and domain index per finite volume."""
        L = [self.L_n, self.L_s, self.L_p]
        eps = [self.eps_n, self.eps_s, self.eps_p]
        brg = [self.b_n, self.b_s, self.b_p]
        tot = sum(L)
        n_k = [max(8, int(round(self.n_fv * Lk / tot))) for Lk in L]
        dx, e_i, d_i, dom = [], [], [], []
        for k, (Lk, nk) in enumerate(zip(L, n_k)):
            dx += [Lk / nk] * nk
            e_i += [eps[k]] * nk
            d_i += [eps[k] ** brg[k]] * nk      # transport efficiency; D_eff = D_e * this
            dom += [k] * nk
        return np.array(dx), np.array(e_i), np.array(d_i), np.array(dom)

    def _build_modes(self):
        """One-time eigendecomposition of the D_e-free operator. Stores modes and projections."""
        from scipy.linalg import eigh

        self._tables = {}                      # transport-property lookup tables, keyed on name
        self._warned_T = False
        dx, eps_i, d_i, dom = self._grid()
        n = len(dx)
        M = np.diag(eps_i * dx)
        K = np.zeros((n, n))
        for i in range(n - 1):
            # harmonic, distance-weighted face transport efficiency: a porosity jump at an
            # interface is a series resistance, not an average
            d_face = (dx[i] + dx[i + 1]) / (dx[i] / d_i[i] + dx[i + 1] / d_i[i + 1])
            coef = 2.0 * d_face / (dx[i] + dx[i + 1])
            K[i, i] -= coef
            K[i, i + 1] += coef
            K[i + 1, i + 1] -= coef
            K[i + 1, i] += coef

        src = np.zeros(n)                      # volumetric source per ampere of cell current
        pre = (1.0 - self.t_plus) / (self.F * self.A)
        src[dom == 0] = pre / self.L_n
        src[dom == 2] = -pre / self.L_p
        B = src * dx                           # integrated over each cell

        lam, phi = eigh(K, M)                  # ascending; lam[-1] == 0 (the constant mode)
        order = np.argsort(lam)[::-1]          # slowest first
        lam, phi = lam[order], phi[:, order]
        if abs(lam[0]) > 1e-6 * abs(lam[1]):
            logger.warning("Electrolyte: leading eigenvalue %.3e is not the conserved zero mode.", lam[0])
        g_all = phi.T @ B
        if abs(g_all[0]) > 1e-8 * np.max(np.abs(g_all)):
            logger.warning("Electrolyte: the conserved mode is excited (g0=%.3e) - salt balance broken.",
                           g_all[0])
        if self.n_modes >= n:
            raise ValueError(f"n_modes={self.n_modes} needs n_fv > {self.n_modes + 1} cells")

        keep = slice(1, 1 + self.n_modes)      # drop the conserved constant mode
        self._dx, self._eps_i, self._dom = dx, eps_i, dom
        self._x = np.cumsum(dx) - 0.5 * dx
        self._lam_hat = lam[keep]              # [1/m2]; the eigenvalue is D_e * lam_hat
        self._phi = phi[:, keep]               # M-orthonormal
        self._g = g_all[keep]                  # phi^T B
        self._lam_hat_all, self._g_all = lam[1:], g_all[1:]      # for the truncation report
        self._phi_all = phi[:, 1:]

        # averaging rows: <c>_domain = w_k . c
        self._w = {}
        for k in (0, 1, 2):
            w = np.where(dom == k, dx, 0.0)
            self._w[k] = w / w.sum()
        logger.info("Electrolyte %s: %d cells, %d modes, D_e=%.2e m2/s, tau_e=%.1f s, "
                    "R_e=%.2f mOhm, steady-state capture %.4f", self.param_set, n, self.n_modes,
                    self.de_ref, self.tau_e(), self.r_ohmic() * 1e3, self.mode_capture())

    # ------------------------------------------------------------ properties
    @property
    def L_tot(self) -> float:
        return self.L_n + self.L_s + self.L_p

    @property
    def x(self) -> np.ndarray:
        """Finite-volume cell centres [m], 0 at the negative current collector."""
        return self._x

    def _prop(self, name: str, func, ref: float, T: float, c: Optional[float]) -> float:
        """Evaluate a PyBaMM transport property at (c_e0, T), through a log-linear lookup table.

        `step` needs D_e(T) on every call and a PyBaMM expression tree costs ~200 us. A plain
        memo on T does not help, because the temperature that reaches `step` in deployment is a
        MEASUREMENT: it drifts by a fraction of a kelvin every sample and never repeats a value,
        so every lookup would miss. Instead the property is tabulated once over T_TABLE and
        interpolated in log space - these are Arrhenius-like, so log(value) is very nearly
        linear in T and 1 K spacing costs ~1e-5 relative error. Passing an explicit `c` bypasses
        the table (that path is per-call anyway)."""
        if func is None:
            return ref
        T = float(T)
        if c is not None:
            return func(float(c), T)
        tab = self._tables.get(name)
        if tab is None:
            lo, hi, step = T_TABLE
            grid = np.arange(lo, hi + 0.5 * step, step)
            vals = np.array([func(self.c_e0, float(t)) for t in grid], dtype=float)
            tab = (grid, np.log(np.maximum(vals, 1e-300)))
            self._tables[name] = tab
        grid, logv = tab
        if not (grid[0] <= T <= grid[-1]) and not self._warned_T:
            logger.warning("Electrolyte: T=%.1f K is outside the tabulated range [%.1f, %.1f] - "
                           "the transport properties are clamped at the endpoint.",
                           T, grid[0], grid[-1])
            self._warned_T = True
        return float(np.exp(np.interp(T, grid, logv)))

    def De(self, T: float = T_REF, c: Optional[float] = None) -> float:
        """Salt diffusivity [m2/s] at (c, T); c defaults to c_e0 (the SPMe linearisation)."""
        return self.de_mult * self._prop('De', self._de_func, self.de_ref, T, c)

    def kappa(self, T: float = T_REF, c: Optional[float] = None) -> float:
        """Ionic conductivity [S/m] at (c, T)."""
        return self._prop('kappa', self._kappa_func, self.kappa_ref, T, c)

    def tdf(self, T: float = T_REF, c: Optional[float] = None) -> float:
        """Thermodynamic factor 1 + dln f / dln c at (c, T)."""
        return self._prop('tdf', self._tdf_func, self.tdf_ref, T, c)

    def tau_e(self, T: float = T_REF) -> float:
        """Slowest electrolyte time constant [s] - the one that aliases into the solid dynamics."""
        return float(-1.0 / (self.De(T) * self._lam_hat[0]))

    def mode_capture(self) -> float:
        """Fraction of the steady-state (c_p - c_n) split reproduced by the retained modes.
        Temperature-independent: D_e cancels between the kept and the full sum."""
        dw = self._w[2] - self._w[0]
        full = float(np.sum(dw @ self._phi_all * (-self._g_all / self._lam_hat_all)))
        kept = float(np.sum(dw @ self._phi * (-self._g / self._lam_hat)))
        return kept / full if full != 0.0 else 1.0

    # ----------------------------------------------------------------- state
    def initial_state(self) -> np.ndarray:
        """Modal coordinates of a uniform electrolyte at c_e0."""
        return np.zeros(self.n_modes)

    def step(self, q: np.ndarray, I: float, dt: float, T: float = T_REF) -> np.ndarray:
        """Advance the modal state by dt [s] at constant cell current I [A] (I > 0 = discharge).

        Exact for the discretised system: q_i <- e^(l dt) q_i + (e^(l dt) - 1)/l g_i I.
        Unconditionally stable, so dt is a modelling choice, not a stability constraint.
        """
        q = np.asarray(q, dtype=float)
        lam = self.De(T) * self._lam_hat                      # [1/s], all < 0
        z = lam * float(dt)
        gain = np.where(np.abs(z) > 1e-12, np.expm1(z) / lam, float(dt))
        return np.exp(z) * q + gain * self._g * float(I)

    def steady(self, I: float, T: float = T_REF) -> np.ndarray:
        """Modal state the profile relaxes to under a sustained current I - the quasi-steady
        limit that the Stage-1 algebraic formula approximates."""
        return -self._g * float(I) / (self.De(T) * self._lam_hat)

    # ------------------------------------------------------------- observers
    def profile(self, q: np.ndarray) -> np.ndarray:
        """Full electrolyte concentration [mol/m3] on `x`."""
        return self.c_e0 + self._phi @ np.asarray(q, dtype=float)

    def averages(self, q: np.ndarray) -> Tuple[float, float, float]:
        """X-averaged (negative, separator, positive) electrolyte concentration [mol/m3]."""
        c = self.profile(q)
        return tuple(float(self._w[k] @ c) for k in (0, 1, 2))

    def salt_error(self, q: np.ndarray) -> float:
        """Relative departure of the total salt inventory from its initial value - identically
        zero in exact arithmetic, since the conserved mode is not in the basis."""
        c = self.profile(q)
        tot = float(np.sum(self._eps_i * self._dx * c))
        tot0 = float(np.sum(self._eps_i * self._dx) * self.c_e0)
        return (tot - tot0) / tot0

    # ---------------------------------------------------------- overpotentials
    def r_ohmic(self, T: float = T_REF) -> float:
        """Electrolyte ohmic resistance [Ohm] - the SPMe leading-order form, with the 1/3
        factors that come from the linearly varying current inside each electrode."""
        k = self.kappa(T)
        kn, ks, kp = k * self.eps_n ** self.b_n, k * self.eps_s ** self.b_s, k * self.eps_p ** self.b_p
        return (self.L_n / (3.0 * kn) + self.L_s / ks + self.L_p / (3.0 * kp)) / self.A

    def eta_ohmic(self, I: float, T: float = T_REF) -> float:
        """Electrolyte ohmic contribution to the terminal voltage [V] (negative on discharge)."""
        return -float(I) * self.r_ohmic(T)

    def eta_conc(self, q: np.ndarray, T: float = T_REF) -> float:
        """Concentration overpotential [V]:

            eta = (2RT/F)(1 - t_+) int_{ln c_n}^{ln c_p} TDF(c) dln c

        'ref' freezes TDF at c_e0, giving the closed form (2RT/F)(1-t_+) TDF ln(c_p/c_n);
        'quad' evaluates it along the path with 5-point Gauss-Legendre, which recovers most of
        the residual against PyBaMM where TDF varies strongly with concentration.
        """
        c_n, _, c_p = self.averages(q)
        c_n, c_p = max(c_n, 1e-6), max(c_p, 1e-6)
        pre = 2.0 * self.R_gas * float(T) / self.F * (1.0 - self.t_plus)
        if self.tdf_mode == "quad":
            u, w = np.polynomial.legendre.leggauss(5)
            a, b = np.log(c_n), np.log(c_p)
            cs = np.exp(0.5 * (b - a) * u + 0.5 * (b + a))
            integral = 0.5 * (b - a) * float(np.sum(w * np.array([self.tdf(T, c) for c in cs])))
            return pre * integral
        return pre * self.tdf(T) * float(np.log(c_p / c_n))

    def overpotential(self, q: np.ndarray, I: float, T: float = T_REF) -> dict:
        """Both electrolyte contributions to the terminal voltage [V], plus the X-averaged
        concentrations that the exchange-current density needs."""
        c_n, c_s, c_p = self.averages(q)
        ohm, conc = self.eta_ohmic(I, T), self.eta_conc(q, T)
        return {"ohmic": ohm, "conc": conc, "total": ohm + conc,
                "c_e_n": c_n, "c_e_s": c_s, "c_e_p": c_p}

    # -------------------------------------------------------------- manifest
    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        d.update({"L_tot": self.L_tot, "tau_e_ref_s": self.tau_e(),
                  "r_ohmic_ref_ohm": self.r_ohmic(), "mode_capture": self.mode_capture(),
                  "lam_hat": self._lam_hat.tolist(),
                  "tau_modes_ref_s": (-1.0 / (self.De() * self._lam_hat)).tolist()})
        return d

    def __repr__(self):
        return (f"Electrolyte({self.param_set}, n_modes={self.n_modes}, "
                f"D_e={self.de_ref:.2e} m2/s, tau_e={self.tau_e():.0f} s, "
                f"R_e={self.r_ohmic() * 1e3:.2f} mOhm)")
