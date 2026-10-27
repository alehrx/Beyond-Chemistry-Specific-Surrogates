"""
multislice.py - "DFN-lite": N single-particle slices per electrode along the through-cell
coordinate x, sharing ONE solid potential per electrode and coupled through the modal
electrolyte.

Per time step, for each electrode:

    j_k  = 2 j0_k(c_e,k, x_surf,k) sinh( F eta_k / (2 R T) ),  eta_k = phi_s - phi_e,k - U(x_surf,k)
    sum_k j_k A_k = I                                          (charge conservation)

The unknowns (j_n, j_p, phi_s_n, phi_s_p) are solved together by Newton with the exact Jacobian:
phi_e is linear in the reaction currents at frozen concentration, so its response matrix is built
once per step. Fixed-point iteration is NOT enough - it diverges whenever the electrolyte
resistance is large against the charge-transfer resistance, which is the Ai2020 case.

This is a DFN minus the solid-phase potential gradient (kept as a lumped L/3sigma resistance).
N = 1 collapses to the SPMe in `cell.py`, reproducing it to 0.1 mV at 5C.

The electrolyte operator is state-dependent (`de_local=True`): K(c) is reassembled each step from
the local D_e(c_i) with the same harmonic face average PyBaMM applies to a flux, then projected
onto the fixed modal basis, A(c) = Phi^T K(c) Phi. With D_e uniform this reduces exactly to
D_e * diag(lam_hat), i.e. the constant-D_e model, so `de_local=False` recovers it. Salt
conservation is structural either way, since Phi is M-orthogonal to the constant mode.

Accuracy against a PyBaMM DFN (Ai2020, 25 C, N = 5, dt = 20 s), voltage RMSE over V > 3.1 V:

    1C 1.9 mV | 2C 2.5 mV | 3C 3.9 mV | 5C 9.3 mV        (SPMe, N = 1: 10 / 41 / 61 / 598 mV)

Caveat: the slice model needs the surrogate to be valid at the LOCAL flux, not the cell average.
At 5C the peak slice reaches |Phi| = 1.8 (N = 5) and 3.7 (N = 20) against a trained limit of 3.0,
so N is bounded above by the training envelope, not only by cost.

Sign convention: I > 0 = discharge; j > 0 = anodic, so on discharge j_n > 0 and j_p < 0.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.linalg import expm

from .chemistry import Chemistry, T_REF
from .surrogate import Surrogate
from .electrolyte import Electrolyte
from .voltage import CellVoltage

logger = logging.getLogger(__name__)


@dataclass
class SliceState:
    """x_n, x_p: (N_slices, n_sensors) stoichiometry profiles. q: electrolyte modal state."""
    x_n: np.ndarray
    x_p: np.ndarray
    q: np.ndarray
    j_n: Optional[np.ndarray] = None            # last solved reaction currents [A/m2] (warm start)
    j_p: Optional[np.ndarray] = None

    def copy(self):
        return SliceState(self.x_n.copy(), self.x_p.copy(), self.q.copy(),
                          None if self.j_n is None else self.j_n.copy(),
                          None if self.j_p is None else self.j_p.copy())


class MultiSlice:
    """N parallel SPM slices per electrode on the electrolyte's own finite-volume grid."""

    def __init__(self, surrogate: Surrogate, chem_n: Chemistry, chem_p: Chemistry,
                 electrolyte: Electrolyte, voltage: CellVoltage, n_slices: int = 5,
                 n_slices_p: Optional[int] = None, de_local: bool = True):
        self.sur = surrogate
        self.chem = {'n': chem_n, 'p': chem_p}
        self.ely = electrolyte
        self.volt = voltage
        self.param_set = electrolyte.param_set
        self.de_local = bool(de_local)        # D_e(c) in the operator, or frozen at c_e0
        self.N = {'n': int(n_slices), 'p': int(n_slices if n_slices_p is None else n_slices_p)}
        self.F, self.Rg = self.ely.F, self.ely.R_gas
        self.A = self.ely.A
        self.c_e0, self.t_plus = self.ely.c_e0, self.ely.t_plus
        self.r_series = self.volt.r_contact
        self._tab = {}
        self._build_maps()

    @classmethod
    def from_run(cls, run_dir: str, param_set: str, n_slices: int = 5,
                 n_slices_p: Optional[int] = None, n_modes: int = 8, n_fv: int = 150,
                 de_scale="auto", de_local: bool = True, device: str = "cpu") -> "MultiSlice":
        """Build from a trained surrogate run directory and a PyBaMM parameter set."""
        ely = Electrolyte.from_pybamm(param_set, n_modes=n_modes, n_fv=n_fv, de_scale=de_scale)
        return cls(Surrogate.from_run(run_dir, device=device),
                   Chemistry.from_pybamm(param_set, "Negative"),
                   Chemistry.from_pybamm(param_set, "Positive"),
                   ely, CellVoltage.from_pybamm(param_set, electrolyte=ely),
                   n_slices=n_slices, n_slices_p=n_slices_p, de_local=de_local)

    # --------------------------------------------------------------- geometry
    def _build_maps(self):
        """Slice to finite-volume map, the per-slice modal input matrix G, and averaging rows."""
        e = self.ely
        dx, _, d_eff, dom = e._grid()
        self.dx, self.d_eff, self.dom, self.x = dx, d_eff, dom, e._x
        self.maps, cols, raw = {}, [], []
        for key, d in (('n', 0), ('p', 2)):
            idx = np.flatnonzero(dom == d)
            s = np.cumsum(dx[idx]) - 0.5 * dx[idx]                  # centre, local to the electrode
            edges = np.linspace(0.0, dx[idx].sum(), self.N[key] + 1)
            grp = np.clip(np.searchsorted(edges, s) - 1, 0, self.N[key] - 1)
            pre = (1.0 - self.t_plus) * self.chem[key].a / self.F   # mol/m3/s per A/m2 of j
            cells, w, xc = [], [], []
            for k in range(self.N[key]):
                c = idx[grp == k]
                if c.size == 0:
                    raise ValueError('empty slice %d in electrode %s: raise n_fv' % (k, key))
                b = np.zeros(len(dx))
                b[c] = dx[c] * pre
                cols.append(e._phi.T @ b)
                raw.append(b)
                cells.append(c)
                w.append(dx[c] / dx[c].sum())
                xc.append(float(w[-1] @ self.x[c]))
            self.maps[key] = dict(cells=cells, w=w, x=np.array(xc))
        self.G = np.column_stack(cols)                              # (n_modes, N_n + N_p)
        self.B = np.column_stack(raw)                               # cell-integrated sources
        # salt conservation: a balanced current pair must integrate to zero source over the cell,
        # which is what keeps the excluded conserved mode unexcited
        j_bal = np.r_[np.full(self.N['n'], 1.0 / (self.chem['n'].a * self.chem['n'].L)),
                      np.full(self.N['p'], -1.0 / (self.chem['p'].a * self.chem['p'].L))]
        src = self.B @ j_bal
        self.salt_leak = float(abs(src.sum()) / np.abs(src).sum())
        logger.info('%s: %d + %d slices on %d cells, salt leak %.1e', self.param_set,
                    self.N['n'], self.N['p'], len(dx), self.salt_leak)

    # ----------------------------------------------------------------- tables
    def _tables(self, T: float):
        """U(sto), j0(sto) per electrode and kappa(c), TDF(c) on a grid - built once per T."""
        key = round(float(T), 3)
        if key in self._tab:
            return self._tab[key]
        sto = np.linspace(1e-4, 1.0 - 1e-4, 801)
        tab = dict(sto=sto)
        for k, el in (('n', 'Negative'), ('p', 'Positive')):
            tab['U_' + k] = np.array([self.volt.ocp(el, s, T) for s in sto])
            tab['j0_' + k] = np.array([self.volt.j0(el, s, self.c_e0, T) for s in sto])
        c = np.linspace(0.02 * self.c_e0, 4.0 * self.c_e0, 241)
        tab['c'] = c
        tab['kappa'] = np.array([self.ely._kappa_func(ci, T) for ci in c])
        tab['tdf'] = np.array([self.ely._tdf_func(ci, T) for ci in c])
        tab['de'] = np.array([self.ely._de_func(ci, T) for ci in c])
        # j0 is tabulated at c_e0 and rescaled as sqrt(c_e/c_e0): check that against the function
        err = 0.0
        for s in (0.25, 0.5, 0.75):
            for cc in (0.4 * self.c_e0, 2.0 * self.c_e0):
                ref = self.volt.j0('Negative', s, cc, T)
                app = self.volt.j0('Negative', s, self.c_e0, T) * np.sqrt(cc / self.c_e0)
                err = max(err, abs(app - ref) / max(abs(ref), 1e-12))
        tab['j0_sqrt_err'] = err
        if err > 1e-6:
            logger.warning('j0 is not sqrt(c_e) in %s (max dev %.2e) - tabulate in 2D instead',
                           self.param_set, err)
        self._tab[key] = tab
        return tab

    # ------------------------------------------------------------ electrolyte
    def operator(self, c, T):
        """Galerkin projection of the state-dependent diffusion operator onto the fixed basis.

        Assembles K(c) from the local D_e(c_i) with the same harmonic, distance-weighted face
        average PyBaMM uses for a flux, then returns A = Phi^T K(c) Phi [1/s]. With D_e uniform
        this is exactly D_e * diag(lam_hat): the frozen model is the special case.
        """
        tab = self._tables(T)
        d = self.d_eff * np.interp(c, tab['c'], tab['de'])      # tor * D_e(c) on the nodes
        dx = self.dx
        d_face = (dx[:-1] + dx[1:]) / (dx[:-1] / d[:-1] + dx[1:] / d[1:])
        coef = 2.0 * d_face / (dx[:-1] + dx[1:])                # face conductance
        phi = self.ely._phi
        flux = coef[:, None] * (phi[1:] - phi[:-1])             # K is tridiagonal: assemble K @ Phi
        Kp = np.zeros_like(phi)
        Kp[:-1] += flux
        Kp[1:] -= flux
        return phi.T @ Kp

    def step_modal(self, q, src, dt, T, c=None):
        """Advance the modal state by dt at constant source `src` [modal units per second].

        Frozen operator: n decoupled scalar exponentials, as before. State-dependent operator:
        the modes couple, so use the augmented matrix
            expm([[A, g], [0, 0]] dt) = [[e^(A dt), int_0^dt e^(A s) ds g], [0, 1]]
        which is still exact for a source held constant over the step.
        """
        if self.de_local and c is not None:
            A = self.operator(c, T)
            n = len(q)
            Z = np.zeros((n + 1, n + 1))
            Z[:n, :n] = A * dt
            Z[:n, n] = src * dt
            E = expm(Z)
            return E[:n, :n] @ q + E[:n, n]
        lam = self.ely.De(T) * self.ely._lam_hat
        z = lam * float(dt)
        gain = np.where(np.abs(z) > 1e-12, np.expm1(z) / lam, float(dt))
        return np.exp(z) * q + gain * src

    # ------------------------------------------------------------- potentials
    def _cumint(self, f):
        """Midpoint cumulative integral of f over the grid, starting from phi(x=0) = 0."""
        return np.cumsum(f * self.dx) - 0.5 * f * self.dx

    def _transport(self, c, T):
        """Frozen-concentration electrolyte response: the ohmic matrix P (phi_e per unit j of
        each slice, columns ordered anode then cathode) and the diffusion potential."""
        tab = self._tables(T)
        kap = np.interp(c, tab['c'], tab['kappa']) * self.d_eff
        tdf = np.interp(c, tab['c'], tab['tdf'])
        dlnc = np.gradient(np.log(np.maximum(c, 1.0)), self.x)
        phi_diff = self._cumint((2.0 * self.Rg * T / self.F) * (1.0 - self.t_plus) * tdf * dlnc)
        cols = []
        for key in ('n', 'p'):
            for cells in self.maps[key]['cells']:
                u = np.zeros(len(self.dx))
                u[cells] = self.chem[key].a
                i_e = np.cumsum(u * self.dx) - 0.5 * u * self.dx
                cols.append(self._cumint(-i_e / kap))
        return np.column_stack(cols), phi_diff, kap

    def phi_e(self, j_n, j_p, c, T):
        """Electrolyte potential on the grid [V], phi_e(x=0) = 0: the ohmic and diffusion terms
        of d(phi_e)/dx = -i_e/kappa_eff + (2RT/F)(1-t+) TDF dln(c_e)/dx."""
        P, phi_diff, _ = self._transport(c, T)
        j = np.r_[np.asarray(j_n, dtype=float), np.asarray(j_p, dtype=float)]
        u = np.zeros(len(self.dx))
        for key, jj in (('n', j_n), ('p', j_p)):
            for k, cells in enumerate(self.maps[key]['cells']):
                u[cells] = self.chem[key].a * jj[k]
        i_e = np.cumsum(u * self.dx) - 0.5 * u * self.dx
        return P @ j + phi_diff, i_e

    def _bv(self, j0, U, eta, T):
        """Butler-Volmer current and its derivative wrt the overpotential."""
        z = np.clip((self.F / (2.0 * self.Rg * T)) * eta, -200.0, 200.0)
        return 2.0 * j0 * np.sinh(z), (self.F / (self.Rg * T)) * j0 * np.cosh(z)

    # ------------------------------------------------------------------ solve
    def split(self, st: SliceState, I: float, T: float = T_REF, tol: float = 1e-8,
              itmax: int = 60) -> dict:
        """Solve the current distribution at frozen states.

        Unknowns (j_n, j_p, phi_s_n, phi_s_p); equations Butler-Volmer per slice plus one charge
        balance per electrode. Newton with the exact Jacobian and a backtracking line search -
        the coupling through phi_e is too strong for fixed-point iteration at Ai2020 resistances.
        """
        tab = self._tables(T)
        I = float(I)
        c = self.ely.profile(st.q)
        P, phi_diff, _ = self._transport(c, T)
        idx = {'n': slice(0, self.N['n']), 'p': slice(self.N['n'], self.N['n'] + self.N['p'])}
        nt = self.N['n'] + self.N['p']

        U, j0, Pbar, dbar, tgt = {}, {}, {}, {}, {}
        for k in ('n', 'p'):
            cells, w = self.maps[k]['cells'], self.maps[k]['w']
            sto_s = np.asarray(getattr(st, 'x_' + k))[:, -1]
            c_s = np.array([wi @ c[ci] for wi, ci in zip(w, cells)])
            U[k] = np.interp(sto_s, tab['sto'], tab['U_' + k])
            j0[k] = (np.interp(sto_s, tab['sto'], tab['j0_' + k])
                     * np.sqrt(np.maximum(c_s, 1.0) / self.c_e0))
            Pbar[k] = np.array([wi @ P[ci] for wi, ci in zip(w, cells)])       # (N_k, nt)
            dbar[k] = np.array([wi @ phi_diff[ci] for wi, ci in zip(w, cells)])
            tgt[k] = (1.0 if k == 'n' else -1.0) * I / (self.chem[k].a * self.chem[k].L * self.A)

        def residual(y):
            j, r = y[:nt], np.zeros(nt + 2)
            for m, k in enumerate(('n', 'p')):
                eta = y[nt + m] - (Pbar[k] @ j + dbar[k]) - U[k]
                bv, _ = self._bv(j0[k], U[k], eta, T)
                r[idx[k]] = j[idx[k]] - bv
                r[nt + m] = float(np.mean(j[idx[k]])) - tgt[k]
            return r

        def jacobian(y):
            j = y[:nt]
            J = np.zeros((nt + 2, nt + 2))
            for m, k in enumerate(('n', 'p')):
                eta = y[nt + m] - (Pbar[k] @ j + dbar[k]) - U[k]
                _, D = self._bv(j0[k], U[k], eta, T)
                rows = np.arange(self.N[k]) + (0 if k == 'n' else self.N['n'])
                J[np.ix_(rows, np.arange(nt))] = D[:, None] * Pbar[k]
                J[rows, rows] += 1.0
                J[rows, nt + m] = -D
                J[nt + m, idx[k]] = 1.0 / self.N[k]
            return J

        # initial guess: warm start on j, phi_s from the uniform-eta estimate
        y = np.zeros(nt + 2)
        for k in ('n', 'p'):
            jk = getattr(st, 'j_' + k)
            y[idx[k]] = np.full(self.N[k], tgt[k]) if jk is None else np.asarray(jk, dtype=float)
        for m, k in enumerate(('n', 'p')):
            eta0 = (2.0 * self.Rg * T / self.F) * np.arcsinh(tgt[k] / (2.0 * j0[k]))
            y[nt + m] = float(np.mean(Pbar[k] @ y[:nt] + dbar[k] + U[k] + eta0))

        r = residual(y)
        nrm = float(np.linalg.norm(r))
        it = 0
        for it in range(1, itmax + 1):
            if nrm <= tol * max(1.0, abs(tgt['n'])):
                break
            try:
                d = np.linalg.solve(jacobian(y), -r)
            except np.linalg.LinAlgError:
                logger.warning('singular Jacobian in the current split')
                break
            step = 1.0
            for _ in range(40):                    # backtracking on the residual norm
                y_try = y + step * d
                r_try = residual(y_try)
                n_try = float(np.linalg.norm(r_try))
                if np.isfinite(n_try) and n_try < (1.0 - 1e-4 * step) * nrm:
                    break
                step *= 0.5
            else:
                break                              # no progress: accept the current iterate
            y, r, nrm = y_try, r_try, n_try
        else:
            logger.warning('current split did not converge: |r| = %.2e after %d Newton steps',
                           nrm, itmax)

        j = y[:nt]
        phi_s = {'n': float(y[nt]), 'p': float(y[nt + 1])}
        pe = P @ j + phi_diff
        u = np.zeros(len(self.dx))
        for k in ('n', 'p'):
            for m, cells in enumerate(self.maps[k]['cells']):
                u[cells] = self.chem[k].a * j[idx[k]][m]
        i_e = np.cumsum(u * self.dx) - 0.5 * u * self.dx
        return dict(V=float(phi_s['p'] - phi_s['n'] - I * self.r_series),
                    j_n=j[idx['n']].copy(), j_p=j[idx['p']].copy(), phi_s=phi_s, phi_e=pe,
                    i_e=i_e, c_e=c, sto_n=np.asarray(st.x_n)[:, -1], sto_p=np.asarray(st.x_p)[:, -1],
                    c_e_n=np.array([w @ c[ci] for w, ci in zip(self.maps['n']['w'],
                                                               self.maps['n']['cells'])]),
                    c_e_p=np.array([w @ c[ci] for w, ci in zip(self.maps['p']['w'],
                                                               self.maps['p']['cells'])]),
                    iters=it, residual=float(nrm))

    # ------------------------------------------------------------------ steps
    def step(self, st: SliceState, I: float, dt: float, T: float = T_REF, out=None) -> SliceState:
        """One time step: solve the split at the current state, then advance particles and salt.
        Pass `out` to reuse a split already computed at this state."""
        out = self.split(st, I, T) if out is None else out
        new = st.copy()
        for k in ('n', 'p'):
            chem = self.chem[k]
            # equivalent CELL current of this slice: j_n = +I/(a L A) but j_p = -I/(a L A), and
            # Chemistry.sign already carries the electrode polarity, so undo the cathode sign here
            area = (1.0 if k == 'n' else -1.0) * chem.a * chem.L * self.A
            x = np.asarray(getattr(st, 'x_' + k))
            # every slice of an electrode shares dt and D_s, so the whole electrode is one call
            setattr(new, 'x_' + k, self.sur.step_batch(x, out['j_' + k] * area, dt, chem, T))
            setattr(new, 'j_' + k, out['j_' + k])
        new.q = self.step_modal(st.q, self.G @ np.r_[out['j_n'], out['j_p']], float(dt), T,
                                c=out['c_e'])
        return new

    def initial_state(self, x_n0: float, x_p0: float) -> SliceState:
        n = self.sur.n
        return SliceState(np.full((self.N['n'], n), float(x_n0)),
                          np.full((self.N['p'], n), float(x_p0)),
                          self.ely.initial_state())

    def rollout(self, t: Sequence[float], I, x_n0: float, x_p0: float, T: float = T_REF,
                keep_fields: bool = True) -> dict:
        """Constant-current-per-step rollout. I: scalar or per-step array [A], > 0 = discharge."""
        t = np.asarray(t, dtype=float)
        I = np.full(len(t), float(I)) if np.isscalar(I) else np.asarray(I, dtype=float)
        st = self.initial_state(x_n0, x_p0)
        n_t = len(t)
        V, soc = np.zeros(n_t), np.zeros(n_t)
        jn, jp = np.zeros((n_t, self.N['n'])), np.zeros((n_t, self.N['p']))
        sn = np.zeros((n_t, self.N['n']))
        sp = np.zeros((n_t, self.N['p']))
        ce = np.zeros((n_t, len(self.dx))) if keep_fields else None
        pe = np.zeros((n_t, len(self.dx))) if keep_fields else None
        for k in range(n_t):
            out = self.split(st, I[k], T)
            V[k], jn[k], jp[k], sn[k] = out['V'], out['j_n'], out['j_p'], out['sto_n']
            sp[k] = out['sto_p']
            soc[k] = float(np.mean([self.sur.volume_mean(x) for x in st.x_n]))
            if keep_fields:
                ce[k], pe[k] = out['c_e'], out['phi_e']
            if k + 1 < n_t:
                st = self.step(st, I[k], t[k + 1] - t[k], T, out=out)
        return dict(t=t, I=I, V=V, j_n=jn, j_p=jp, c_e=ce, phi_e=pe, sto_n_surf=sn, sto_p_surf=sp, x_n_mean=soc,
                    x=self.x, x_slice_n=self.maps['n']['x'], x_slice_p=self.maps['p']['x'], state=st)

    def __repr__(self):
        return ('MultiSlice(%s, %d+%d slices, %d modes, de_scale=%g, D_e %s)'
                % (self.param_set, self.N['n'], self.N['p'], self.ely.n_modes,
                   self.ely.de_scale, 'local' if self.de_local else 'frozen'))
