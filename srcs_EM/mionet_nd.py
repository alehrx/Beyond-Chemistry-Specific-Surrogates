"""
mionet_nd.py
============
MIONet v2 - the DIMENSIONLESS single-particle diffusion operator.

Learns the solution operator of the parameter-free problem (see srcs_EM/chemistry.py)

    dx/dtau = xi^-2 d/dxi ( xi^2 dx/dxi ),      -dx/dxi |_{xi=1} = Phi

    branch 1 : x0(xi)  - the initial radial profile sampled at `n_sensors` points
    branch 2 : Phi     - the dimensionless surface flux (scalar)
    trunk    : (tau, xi)

    x(tau, xi) = sum_k  b1_k(x0) * b2_k(Phi) * t_k(tau, xi)  +  b0  +  q * xi^2 / 2

b0 is the trainable scalar output bias of the MIONet / DeepONet papers (output_bias=True; it
measurably helps accuracy there). Legacy manifests without the key load with output_bias=False.

The problem is parameter-free, so there is NO third branch: chemistry, temperature and
electrode all live in the (Phi, tau) conversion (srcs_EM.chemistry) before the net is called.

Hard-BC ansatz (ON by default): psi = xi^2/2 has psi'(0) = 0 (centre symmetry untouched)
and psi'(1) = 1, so with q = -Phi the inhomogeneous surface flux is satisfied EXACTLY by
construction and the soft BC loss only has to drive the homogeneous remainder
dN/dxi|_1 -> 0. In v1 this was q = -sign*i*C/j0_aux; in the dimensionless variables that
is precisely -Phi, so no chemistry constants live in the network any more.

`forward` / `predict` keep the positional signatures of model_old.MIONet
(forward(x1, x2, cs_0, current) / predict(coordinate, input1, input2)) so the existing
Trainer and loss functions can be reused: in v2 the "current" slot simply carries Phi.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init

_ACTS = {'tanh': nn.Tanh, 'silu': nn.SiLU, 'gelu': nn.GELU}


def _mlp(in_dim: int, hidden_dim: int, n_layers: int, act: str) -> nn.Sequential:
    """[Linear -> act] x n_layers, ending with an activation (DeepONet-style feature map)."""
    layers, width = [], in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(width, hidden_dim), _ACTS[act]()]
        width = hidden_dim
    seq = nn.Sequential(*layers)
    for m in seq:
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight); init.zeros_(m.bias)
    return seq


class MIONetND(nn.Module):
    # Extra trunk time features. 'linear' is (tau, xi) as every run so far.
    #
    # 'linear+sqrt' adds sqrt(tau) as a third input. The reason is that for tau << 1 the sphere is
    # locally planar and the surface obeys x_s(tau) - x_s(0) = -2 Phi sqrt(tau/pi) - verified
    # against the finite-volume solution to within 3 % across the whole deployment range. That
    # function has an UNBOUNDED derivative at tau = 0: measured d x_s/d tau is 104 at tau = 1e-4,
    # 15.0 at 0.0024 and 3.5 at 0.1. A smooth tanh MLP in linear tau cannot represent it there,
    # and the two cells that report the worst error in every run query exactly there - the Prada
    # anode at tau = 0.0024 and the Chen cathode at 0.0029, i.e. t = 0.024 and 0.029 of the
    # trained window. In sqrt(t) they sit at 0.155 and 0.170, and the response becomes linear in
    # the coordinate.
    #
    # Both features are passed, not just the square root: the network keeps a smooth path and can
    # down-weight the sharp one, and the PDE residual stays finite because d/dtau of sqrt(tau+eps)
    # is bounded by 1/(2 sqrt(eps)).
    TIME_FEATURES = ('linear', 'linear+sqrt')
    SQRT_EPS = 1e-6

    def __init__(self, n_sensors: int, hidden_dim: int = 128, n_hidden_layers: int = 4,
                 hard_bc: bool = True, activation: str = 'tanh', output_bias: bool = True,
                 time_features: str = 'linear'):
        super().__init__()
        if activation not in _ACTS:
            raise ValueError(f"activation must be one of {list(_ACTS)}, got {activation!r}")
        self.n_sensors, self.hidden_dim = int(n_sensors), int(hidden_dim)
        self.n_hidden_layers = int(n_hidden_layers)
        self.hard_bc, self.activation = bool(hard_bc), activation
        self.output_bias = bool(output_bias)
        if time_features not in self.TIME_FEATURES:
            raise ValueError(f"time_features must be one of {list(self.TIME_FEATURES)}, "
                             f"got {time_features!r}")
        self.time_features = time_features
        n_trunk = 3 if time_features == 'linear+sqrt' else 2

        self.branch1 = _mlp(self.n_sensors, hidden_dim, n_hidden_layers, activation)   # x0(xi)
        self.branch2 = _mlp(1, hidden_dim, n_hidden_layers, activation)                # Phi
        self.trunk = _mlp(n_trunk, hidden_dim, n_hidden_layers, activation)            # (tau[, sqrt], xi)
        if self.output_bias:
            self.b0 = nn.Parameter(torch.zeros(1))                                   # MIONet output bias

    # ------------------------------------------------------------------ config
    @property
    def config(self) -> dict:
        """Everything needed to rebuild the architecture (goes into surrogate_manifest.json)."""
        return dict(n_sensors=self.n_sensors, hidden_dim=self.hidden_dim,
                    n_hidden_layers=self.n_hidden_layers, hard_bc=self.hard_bc, activation=self.activation,
                    output_bias=self.output_bias, time_features=self.time_features)

    @classmethod
    def from_config(cls, cfg: dict) -> "MIONetND":
        cfg = dict(cfg)
        b3 = cfg.pop('branch3_dim', 0)                       # legacy manifests (third branch removed)
        cfg.setdefault('output_bias', False)                 # legacy manifests predate the output bias
        cfg.setdefault('time_features', 'linear')            # legacy manifests predate the sqrt feature
        if b3:
            raise ValueError("this manifest used a third branch (branch3_dim > 0), no longer supported")
        return cls(**cfg)

    # ------------------------------------------------------------------ ansatz
    @staticmethod
    def _q(phi: torch.Tensor) -> torch.Tensor:
        """Hard-BC surface-flux coefficient: -dx/dxi|_1 = Phi  =>  q = -Phi."""
        return -phi

    def _product(self, cs_0, phi):
        return self.branch1(cs_0) * self.branch2(phi)

    def _trunk_in(self, t, xi, dim):
        """Trunk input: (t, xi), plus sqrt(t) when time_features says so. Differentiating through
        the square root is what gives the trunk an unbounded-slope basis function at t -> 0."""
        if self.time_features == 'linear+sqrt':
            return torch.cat((t, torch.sqrt(t.clamp_min(0.0) + self.SQRT_EPS), xi), dim)
        return torch.cat((t, xi), dim)

    # ------------------------------------------------------------------ passes
    def forward(self, x1, x2, cs_0, phi):
        """Training pass. x1=tau, x2=xi: [N, Ny, 1]; cs_0: [N, 1, n_sensors]; phi: [N, 1, 1].
        Returns x(tau, xi): [N, Ny, 1]."""
        prod = self._product(cs_0, phi)                         # [N, 1, h]
        tr = self.trunk(self._trunk_in(x1, x2, 2))              # [N, Ny, h]
        out = torch.sum(prod * tr, 2).unsqueeze(2)              # [N, Ny, 1]
        if self.output_bias:
            out = out + self.b0
        if self.hard_bc:
            out = out + self._q(phi) * (x2 ** 2) * 0.5          # q: [N,1,1] broadcasts over Ny
        return out

    def predict(self, coordinate, input1, input2):
        """Inference pass. coordinate: [M, 2] = (tau, xi); input1: [n_sensors]; input2: [1, 1] (Phi).
        Returns x at the M coordinates: [M]."""
        prod = self._product(input1, input2)                    # [1, h] (or [h])
        tr = self.trunk(self._trunk_in(coordinate[:, :1], coordinate[:, 1:2], 1))   # [M, h]
        out = torch.sum(prod * tr, 1)                           # [M]
        if self.output_bias:
            out = out + self.b0
        if self.hard_bc:
            xi = coordinate[:, 1]
            out = out + self._q(input2.reshape(-1)[0]) * (xi ** 2) * 0.5
        return out
