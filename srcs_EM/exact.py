"""Exact solution of the dimensionless particle problem, for chemistry-free validation.

    dx/dtau = (1/r^2) d/dr (r^2 dx/dr),   dx/dr(0) = 0,   dx/dr(1) = -Phi,   x(0, r) = x0(r)

with constant Phi over the step. Writing <x> = 3 int_0^1 x r^2 dr for the particle mean,

    x(tau, r) = <x0> - 3 Phi tau - Phi (r^2/2 - 3/10) + sum_n a_n exp(-lam_n^2 tau) phi_n(r)

  * <x0> - 3 Phi tau      the mean, which falls exactly as the flux dictates (the zero mode)
  * -Phi (r^2/2 - 3/10)   the zero-mean quasi-steady profile that carries the flux
  * phi_n = sin(lam_n r) / (lam_n r), tan(lam_n) = lam_n: the Neumann eigenfunctions, orthogonal
    with weight r^2; a_n projects w0 = x0 - <x0> + Phi (r^2/2 - 3/10) onto them.

The profile is carried on a fine uniform grid and the projections use Simpson's rule with the r^2
weight. Truncating at `n_modes` drops terms below exp(-lam_N^2 tau); for the default 120 modes
(lam_N ~ 380) that is < 1e-60 for tau >= 1e-3, so the only approximation is the quadrature.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def eigenvalues(n: int) -> np.ndarray:
    """The first n positive roots of tan(lam) = lam: one in each (k pi, k pi + pi/2), k >= 1."""
    f = lambda x: np.sin(x) - x * np.cos(x)               # same roots, no poles
    return np.array([brentq(f, k * np.pi + 1e-9, k * np.pi + np.pi / 2 - 1e-9)
                     for k in range(1, n + 1)])


class SphereExact:
    def __init__(self, n_modes: int = 120, n_fine: int = 2001):
        if n_fine % 2 == 0:
            raise ValueError('n_fine must be odd for Simpson weights')
        self.r = np.linspace(0.0, 1.0, n_fine)
        h = self.r[1] - self.r[0]
        w = np.ones(n_fine); w[1:-1:2] = 4.0; w[2:-1:2] = 2.0
        self.wq = w * h / 3.0 * self.r ** 2                  # Simpson x r^2: int f r^2 dr = f @ wq
        self.lam = eigenvalues(n_modes)
        lr = np.outer(self.lam, self.r)
        with np.errstate(invalid='ignore', divide='ignore'):
            B = np.where(lr > 0, np.sin(lr) / np.where(lr > 0, lr, 1.0), 1.0)
        self.B = B                                           # (n_modes, n_fine)
        self.norm = (B ** 2) @ self.wq                      # <phi_n, phi_n>
        self.P = (B * self.wq) / self.norm[:, None]         # projection: a = P @ w0
        self.g = self.r ** 2 / 2.0 - 0.3                     # zero-mean quasi-steady shape

    def mean(self, x: np.ndarray) -> np.ndarray:
        return 3.0 * (np.asarray(x) @ self.wq)

    def step(self, x: np.ndarray, phi, tau) -> np.ndarray:
        """Advance profiles x (B, n_fine) by tau (B,) at constant flux phi (B,)."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        phi = np.broadcast_to(np.asarray(phi, dtype=float), (x.shape[0],))[:, None]
        tau = np.broadcast_to(np.asarray(tau, dtype=float), (x.shape[0],))[:, None]
        m0 = self.mean(x)[:, None]
        w0 = x - m0 + phi * self.g
        a = w0 @ self.P.T                                    # (B, n_modes)
        w = (a * np.exp(-self.lam[None, :] ** 2 * tau)) @ self.B
        return m0 - 3.0 * phi * tau - phi * self.g + w

    def to_fine(self, x_nodes: np.ndarray, nodes: np.ndarray) -> np.ndarray:
        """Piecewise-linear profile through node values, on the fine grid: the function the
        network's branch input stands for."""
        x_nodes = np.atleast_2d(x_nodes)
        return np.stack([np.interp(self.r, nodes, v) for v in x_nodes])

    def at(self, x: np.ndarray, nodes: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return np.stack([np.interp(nodes, self.r, v) for v in x])
