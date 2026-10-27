"""
surrogate.py
============
The ONE deployment API for the v2 dimensionless MIONet.

    sur  = Surrogate.from_run("runs/v2_nondim/trained_model_s42_lbfgs")   # self-describing artifact
    chem = Chemistry.from_pybamm("Mohtat2020", "Negative")
    x1   = sur.step(x0, I=-10.0, dt=60.0, chem=chem, T=268.15)      # (n_sensors,) stoichiometry

`step` does the whole dimensional -> dimensionless -> network -> dimensional chain:
  1. (I, dt, T) -> (Phi, tau) via Chemistry (all chemistry + temperature live there),
  2. centre-pinning: the network was trained on profiles with x(xi=0, 0) = 0, and the
     PDE is linear, so x(tau, xi) = x0[0] + y(tau, xi) with y0 = x0 - x0[0] (exactly what
     the legacy observer F() did),
  3. SUB-STEPPING: if tau exceeds the trained tau_max the step is chained in windows of
     tau_max (the operator is autonomous for a constant Phi), so a hot cell with a large
     tau per window can never push the network outside its trained time range,
  4. an envelope check on |Phi| (flux cannot be sub-stepped): a warning is logged once,
  5. MASS CORRECTION (mass_correction=True): after every window the profile is shifted by a
     constant so that the volume mean <x> = 3*int(x xi^2 dxi) of the stored state changes by
     EXACTLY -3*Phi*dtau (the analytic lithium balance of the sphere). The network's small
     per-step mass bias (particle_001: +4.5e-3 per step, 25 % of the true change) otherwise
     integrates into a rollout drift (-0.46 vs -0.84 after 60 steps at 25 C). The mean is
     taken on the piecewise-linear interpolant of the sensor values for input and output
     alike, so the balance telescopes exactly over any rollout; a constant shift is the
     minimum-norm correction in the xi^2-weighted L2 norm.

No pickle: weights are loaded with torch.load(weights_only=True) and the architecture is
rebuilt from surrogate_manifest.json via MIONetND.from_config.
"""
from __future__ import annotations

import os
import json
import logging

import numpy as np
import torch

from .mionet_nd import MIONetND
from .chemistry import Chemistry, T_REF

logger = logging.getLogger(__name__)


class Surrogate:
    TRUNK_CACHE_MAX = 256          # distinct tau values kept; see _trunk

    def __init__(self, net: MIONetND, manifest: dict, run_dir: str = "", device: str = "cpu",
                 mass_correction: bool = True, substep_policy: str = "equal"):
        self.net = net.to(device).eval()
        self.mass_correction = bool(mass_correction)
        if substep_policy not in ("equal", "legacy"):
            raise ValueError("substep_policy must be 'equal' or 'legacy'")
        self.substep_policy = substep_policy
        self.mass_shift_last = 0.0                                            # diagnostic (last window)
        self.manifest = manifest
        self.run_dir = run_dir
        self.device = device
        self.sensors = np.asarray(manifest["sensors"], dtype=float)         # xi grid of the profile
        self.n = len(self.sensors)
        dl = manifest["dimensionless"]
        self.tau_max = float(dl["tau_range"][1])
        self.phi_max = float(dl["phi_range_recommended"][1])
        self._warned_phi = False
        self.n_substeps_last = 0                                              # diagnostic
        self._xi_q = np.linspace(0.0, 1.0, 401)                               # quadrature grid for <x>
        self._w_q = self._xi_q ** 2
        self._trunk_cache: dict = {}                                          # see _trunk
        self._np = None                                                       # see use_numpy
        self._xi2_half = torch.tensor(self.sensors ** 2 * 0.5, dtype=torch.float32,
                                      device=device)                          # hard-BC ansatz

    # ------------------------------------------------------------------ load
    @classmethod
    def from_run(cls, run_dir: str, device: str = "cpu", mass_correction: bool = True,
                 substep_policy: str = "equal") -> "Surrogate":
        """Load model.pt (state_dict only) + surrogate_manifest.json from a v2 run folder."""
        with open(os.path.join(run_dir, "surrogate_manifest.json"), "r", encoding="utf-8") as f:
            manifest = json.load(f)
        net = MIONetND.from_config(manifest["architecture"])
        sd = torch.load(os.path.join(run_dir, manifest.get("weights_file", "model.pt")),
                        map_location="cpu", weights_only=True)
        net.load_state_dict(sd)
        logger.info("Surrogate %s loaded from %s (n_sensors=%d, tau_max=%.3f, |Phi|<=%.2f)",
                    manifest.get("version", "?"), run_dir, len(manifest["sensors"]),
                    manifest["dimensionless"]["tau_range"][1], manifest["dimensionless"]["phi_range_recommended"][1])
        return cls(net, manifest, run_dir, device, mass_correction=mass_correction,
                   substep_policy=substep_policy)

    def substeps(self, tau: float) -> list:
        """Split a step of `tau` into windows the network can take, none longer than tau_max.

        'equal' (default): ceil(tau/tau_max) windows of tau/ceil(...) each.
        'legacy': floor(tau/tau_max) full windows plus the remainder, and a zero-length step
        evaluated at tau = 0 instead of returned unchanged - what this class did before.

        The remainder is the problem. It can be arbitrarily small, and a near-zero window is the
        worst thing to ask this network for: the true change is then tiny while the network's error
        is not, so the relative error is largest exactly there, and the legacy rule appends one such
        call to every step that overshoots tau_max. Measured over the headline grid, equal windows cut the error by up
        to 3.3x wherever a step exceeds tau_max, never lose anywhere, and never cost an extra call.
        """
        tau = float(tau)
        if tau <= 1e-12:
            return [0.0] if self.substep_policy == "legacy" else []
        if self.substep_policy == "equal":
            n = int(np.ceil(tau / self.tau_max))
            return [tau / n] * n
        n_full, rem = divmod(tau, self.tau_max)
        return [self.tau_max] * int(n_full) + ([rem] if rem > 1e-12 else [])

    @property
    def vm_weights(self) -> np.ndarray:
        """w with <y> = w . y, exactly, for the piecewise-linear interpolant of the sensor values.

        The volume mean is a LINEAR functional of the node values, so the quadrature is a fixed
        vector: recovering it once from the unit basis turns every later mean into a dot product
        instead of a 401-point interpolation. Identical arithmetic, and it lets a whole block of
        reported profiles be projected with one matrix-vector product."""
        if getattr(self, "_vm_w", None) is None:
            e = np.eye(self.n)
            self._vm_w = np.array([self.volume_mean(e[k]) for k in range(self.n)])
        return self._vm_w

    def volume_mean(self, y: np.ndarray) -> float:
        """<y> = int(y xi^2 dxi) / int(xi^2 dxi) of the piecewise-linear interpolant of the sensor values."""
        yq = np.interp(self._xi_q, self.sensors, np.asarray(y, dtype=float))
        return float(np.trapz(yq * self._w_q, self._xi_q) / np.trapz(self._w_q, self._xi_q))

    # ---------------------------------------------------------------- helpers
    def envelope_check(self, phi: float, tau: float):
        """(ok, message). tau beyond tau_max is handled by sub-stepping, so only |Phi| can
        genuinely leave the trained envelope."""
        if abs(phi) > self.phi_max:
            return False, (f"|Phi|={abs(phi):.3f} exceeds the trained flux range {self.phi_max:.2f}: "
                           f"the network is extrapolating in flux (colder/faster than the training set)")
        return True, "ok"

    def resample(self, profile: np.ndarray, n_target: int) -> np.ndarray:
        """Linear resample of a radial profile onto this surrogate's sensor grid (explicit, not
        hidden). The INPUT is assumed uniform in xi (it comes from outside, at some arbitrary
        radial resolution); the TARGET is self.sensors, which is only linspace(0,1,n) when the
        network was trained on the uniform mesh."""
        profile = np.asarray(profile, dtype=float)
        if len(profile) == n_target:
            return profile
        target = self.sensors if n_target == self.n else np.linspace(0, 1, n_target)
        return np.interp(target, np.linspace(0, 1, len(profile)), profile)

    # ------------------------------------------------------- numpy inference
    # At deployment size this network is DISPATCH-bound, not arithmetic-bound: 16x64x5 is 30 tiny
    # ATen launches (15 Linear + 15 activation) across three MLPs, each costing more in Python and
    # dispatch than the ~300k MACs it performs. Measured on one core, the same maths in numpy is
    # ~3x faster, and with the trunk memoised ~6x. Nothing about the model changes - these are the
    # trained weights, evaluated in the same order, in float32.
    _NP_ACTS = {
        'tanh': np.tanh,
        'silu': lambda z: z / (1.0 + np.exp(-z)),
        'gelu': lambda z: 0.5 * z * (1.0 + __import__('scipy.special', fromlist=['erf']).erf(
            z / np.sqrt(np.float32(2.0)))),
    }

    def use_numpy(self, enable: bool = True) -> "Surrogate":
        """Run inference through numpy instead of PyTorch. Same weights, same result."""
        if not enable:
            self._np = None
            return self
        act = self._NP_ACTS.get(self.net.activation)
        if act is None:
            raise ValueError(f'no numpy activation for {self.net.activation!r}')

        def layers(mlp):
            return [(m.weight.detach().numpy().astype(np.float32).T.copy(),
                     m.bias.detach().numpy().astype(np.float32))
                    for m in mlp if isinstance(m, torch.nn.Linear)]

        self._np = dict(
            act=act, b1=layers(self.net.branch1), b2=layers(self.net.branch2),
            tr=layers(self.net.trunk),
            b0=(float(self.net.b0.detach()) if self.net.output_bias else 0.0),
            sqrt_t=(self.net.time_features == 'linear+sqrt'),
            eps=float(self.net.SQRT_EPS),
            sensors=self.sensors.astype(np.float32),
            xi2_half=(self.sensors.astype(np.float32) ** 2) * np.float32(0.5),
        )
        self._trunk_cache.clear()          # cached torch tensors are the wrong type now
        return self

    def _np_mlp(self, z, W):
        act = self._np['act']
        for A, b in W:
            z = act(z @ A + b)
        return z

    def _np_trunk(self, t_col: float):
        tr = self._trunk_cache.get(t_col)
        if tr is None:
            if len(self._trunk_cache) >= self.TRUNK_CACHE_MAX:
                self._trunk_cache.clear()
            p = self._np
            s = p['sensors']
            t = np.full(len(s), t_col, dtype=np.float32)
            cols = ((t, np.sqrt(max(t_col, 0.0) + p['eps'], dtype=np.float32) * np.ones_like(t), s)
                    if p['sqrt_t'] else (t, s))
            tr = self._trunk_cache[t_col] = self._np_mlp(np.stack(cols, axis=1), p['tr'])
        return tr

    def _one_window_np(self, y0: np.ndarray, phi: float, tau: float) -> np.ndarray:
        p = self._np
        tr = self._np_trunk(tau / self.tau_max)
        prod = (self._np_mlp(np.asarray(y0, dtype=np.float32), p['b1'])
                * self._np_mlp(np.float32(phi).reshape(1), p['b2']))
        out = tr @ prod + p['b0']
        if self.net.hard_bc:                                   # q = -Phi, psi = xi^2/2
            out = out - np.float32(phi) * p['xi2_half']
        return out.astype(float)

    @torch.no_grad()
    def _trunk(self, t_col: float) -> torch.Tensor:
        """Trunk output at (t_col, sensors), memoised on t_col.

        The trunk is 89 % of the forward pass's arithmetic (264k of 298k MACs at 16x64x5) and its
        input is (tau/tau_max, xi): the sensors NEVER move, so for any tau that repeats the whole
        thing is a constant being recomputed. A roll-out with a constant D_s and a uniform sampling
        period visits only a handful of distinct tau - 5 over a 156-step Mohtat protocol - so this
        collapses the forward pass to the two branches and one matrix-vector product.

        It is exact memoisation keyed on the tau actually seen, so it changes no result. Where tau
        varies every step (a thermal model, or D_s(x) with ds_mode='mean') it simply never hits and
        costs one dict lookup.
        """
        tr = self._trunk_cache.get(t_col)
        if tr is None:
            if len(self._trunk_cache) >= self.TRUNK_CACHE_MAX:
                self._trunk_cache.clear()                 # bounded, unlike a plain memo
            coord = torch.tensor(np.column_stack([np.full(self.n, t_col), self.sensors]),
                                 dtype=torch.float32, device=self.device)
            tr = self._trunk_cache[t_col] = self.net.trunk(
                self.net._trunk_in(coord[:, :1], coord[:, 1:2], 1))
        return tr

    @torch.no_grad()
    def _one_window(self, y0: np.ndarray, phi: float, tau: float) -> np.ndarray:
        """Advance a centre-pinned profile by 0 <= tau <= tau_max at constant Phi."""
        if self._np is not None:
            return self._one_window_np(y0, phi, tau)
        tr = self._trunk(tau / self.tau_max)
        in1 = torch.tensor(y0, dtype=torch.float32, device=self.device)
        in2 = torch.tensor([[phi]], dtype=torch.float32, device=self.device)
        out = torch.sum(self.net._product(in1, in2) * tr, 1)
        if self.net.output_bias:
            out = out + self.net.b0
        if self.net.hard_bc:                              # q = -Phi, psi = xi^2/2
            out = out + self.net._q(in2.reshape(-1)[0]) * self._xi2_half
        return out.cpu().numpy().astype(float)

    # ------------------------------------------------------------------ steps
    def step_dimless(self, x0: np.ndarray, phi: float, tau: float) -> np.ndarray:
        """x0 (n_sensors,) stoichiometry on the sensor grid -> profile after tau at constant Phi."""
        x0 = np.asarray(x0, dtype=float)
        if len(x0) != self.n:
            x0 = self.resample(x0, self.n)
        ok, msg = self.envelope_check(phi, tau)
        if not ok and not self._warned_phi:
            logger.warning("Surrogate: %s", msg); self._warned_phi = True
        c0 = float(x0[0])
        y = x0 - c0                                                # centre-pinned IC
        steps = self.substeps(tau)
        if not steps:
            return x0                                          # a zero-length step is the identity
        self.n_substeps_last = len(steps)
        for dtau in steps:
            y_new = self._one_window(y, phi, dtau)
            if self.mass_correction:                               # exact lithium balance of the sphere
                shift = (-3.0 * phi * dtau) - (self.volume_mean(y_new) - self.volume_mean(y))
                y_new = y_new + shift
                self.mass_shift_last = float(shift)
            y = y_new
        return y + c0

    def plan_window(self, chem: Chemistry, t, I, T: float = T_REF, sto=None,
                    tau_target=None, jump_frac: float = 0.05, max_samples: int = 0) -> int:
        """How many of the upcoming samples to fold into ONE network call.

        `t` are the timestamps from now on (t[0] is now), `I` the current held over each of them.
        Returns n >= 1: fold t[0]..t[n] into one call at the mean of I[:n].

        The rule is: keep swallowing samples until either the accumulated tau would pass
        `tau_target` (default tau_max) or the current changes by more than `jump_frac` of the peak,
        whichever comes first. Both halves matter and for different reasons.

        tau is why: the network is least accurate per call at small tau - the true change is tiny
        while its error is not - and the error then accumulates over more calls. Driving it at one
        call per BMS sample is the worst operating point measured anywhere in this work (a DST log
        at 1 Hz: 8.2 % of range, against 0.47 % under this rule at a twentieth of the calls).

        The current is why not: a window averaged across a discontinuity is fed a current that
        never flowed. On DST that single mistake costs 45x - it is a correctness condition, not a
        tuning knob. Under CV the current decays smoothly every sample and averaging it is both
        harmless and the point, which is what `jump_frac` separates.

        tau_target is deliberately below tau_max by default: measured over 60 CC-CV cases, 0.03 is
        better than 0.2 on both mean (0.54 vs 0.67 % of range) and worst case (2.07 vs 4.03),
        because a window that just fits under tau_max leaves the next one a runt. On a 1 Hz log it
        makes no difference at all - the current always changes first.
        """
        t = np.asarray(t, dtype=float)
        I = np.asarray(I, dtype=float)
        n_max = min(len(t) - 1, len(I))
        if max_samples:
            n_max = min(n_max, int(max_samples))
        if n_max <= 1:
            return max(n_max, 1)
        tgt = self.tau_max * 0.15 if tau_target is None else float(tau_target)
        tol = jump_frac * float(np.abs(I[:n_max]).max())
        n = 1
        while n < n_max:
            if abs(I[n] - I[n - 1]) > tol:                      # ... or until the current changes
                break
            if chem.fourier(float(t[n + 1] - t[0]), T, sto) > tgt:
                break
            n += 1
        return n

    @torch.no_grad()
    def window_profiles(self, y0: np.ndarray, phi: float, taus, xi=None) -> np.ndarray:
        """Profiles at MANY times inside ONE window, for the price of one branch evaluation.

        This is the property a finite-volume solver does not have. The branches see only (x0, Phi),
        which are constant across the window, so branch1(x0) * branch2(Phi) is computed ONCE; tau
        enters solely through the trunk, so every extra output time is one more row of a single
        batched trunk call. A time-marching solver has no such separation: to report at 1 s it must
        actually take the 1 s steps.

        `taus` may exceed tau_max; each is still a single evaluation, but beyond the trained range
        the network is extrapolating in time and the caller should sub-step instead.

        Returns (len(taus), len(xi)) - raw network output, centre-pinned like the input, with no
        mass projection: the projection is a per-STATE-step correction and these are reports.
        """
        xi = self.sensors if xi is None else np.asarray(xi, dtype=float)
        taus = np.atleast_1d(np.asarray(taus, dtype=float))
        n_t, n_x = len(taus), len(xi)
        tt = np.repeat(taus / self.tau_max, n_x)                 # (n_t*n_x,)
        xx = np.tile(xi, n_t)
        coord = torch.tensor(np.column_stack([tt, xx]), dtype=torch.float32, device=self.device)
        in1 = torch.tensor(np.asarray(y0, dtype=float), dtype=torch.float32, device=self.device)
        in2 = torch.tensor([[phi]], dtype=torch.float32, device=self.device)
        net = self.net
        prod = net._product(in1, in2)                             # the branches: ONCE
        tr = net.trunk(net._trunk_in(coord[:, :1], coord[:, 1:2], 1))
        out = torch.sum(prod * tr, 1)
        if net.output_bias:
            out = out + net.b0
        if net.hard_bc:
            out = out + net._q(in2.reshape(-1)[0]) * (coord[:, 1] ** 2) * 0.5
        return out.cpu().numpy().astype(float).reshape(n_t, n_x)

    def step_series(self, x0: np.ndarray, I: float, dt: float, chem: Chemistry, T: float = T_REF,
                    ds_mode: str = 'ref', fracs=()) -> tuple:
        """One step, plus a report of the profile at each fraction of it in `fracs` (0 < f < 1).

        tau is a continuous trunk input, so an intermediate report is one extra trunk evaluation
        off the SAME window. It never feeds back into the state, so it cannot accumulate - which is
        what makes a long step free. Take the step at whatever tau the network is most accurate at
        and report at every measurement time inside it, instead of stepping once per measurement.

        Over the headline grid this removes essentially the whole cost of
        a long step: the Prada cathode at 20 samples per step goes from 2.24 % of range to 1.00 %,
        the Chen anode from 1.66 % to 0.63 %, both at 1/20 of the state updates.

        Returns (x_end, [profile at each fraction])."""
        x0 = np.asarray(x0, dtype=float)
        if len(x0) != self.n:
            x0 = self.resample(x0, self.n)
        phi, tau = chem.to_dimless(I, dt, T, self._sto(x0, ds_mode))
        steps = self.substeps(tau)
        if not steps:
            return x0, [x0.copy() for _ in fracs]
        c0 = float(x0[0])
        y = x0 - c0
        want = sorted(float(f) * float(tau) for f in fracs)
        out, done, wi = [], 0.0, 0
        for dtau in steps:
            def _at(t_win, _y=y):
                z = self._one_window(_y, phi, t_win)
                if self.mass_correction:
                    z = z + ((-3.0 * phi * t_win) - (self.volume_mean(z) - self.volume_mean(_y)))
                return z
            while wi < len(want) and want[wi] <= done + dtau + 1e-12:
                out.append(_at(max(want[wi] - done, 0.0)) + c0)
                wi += 1
            y = _at(dtau)
            done += dtau
        out.extend([y + c0] * (len(want) - wi))
        return y + c0, out

    def step_series_at(self, x0: np.ndarray, I: float, dt: float, chem: Chemistry, xi_out,
                       T: float = T_REF, ds_mode: str = 'ref', fracs=()) -> tuple:
        """`step_series`, with the reported profiles evaluated by the network directly at the radii
        `xi_out` instead of on the sensors. The state carried forward stays on the sensors (it is the
        branch input of the next step); the reports at xi_out get the same lithium-balance shift as
        the state, so the two describe the same profile.

        Returns (x_end on the sensors, x_end at xi_out, [profile at xi_out at each fraction])."""
        x0 = np.asarray(x0, dtype=float)
        if len(x0) != self.n:
            x0 = self.resample(x0, self.n)
        xi_out = np.asarray(xi_out, dtype=float)
        phi, tau = chem.to_dimless(I, dt, T, self._sto(x0, ds_mode))
        steps = self.substeps(tau)
        c0 = float(x0[0])
        y = x0 - c0
        if not steps:
            at = np.interp(xi_out, self.sensors, x0)
            return x0, at, [at.copy() for _ in fracs]
        want = sorted(float(f) * float(tau) for f in fracs)
        out, done, wi = [], 0.0, 0
        y_out = None
        for dtau in steps:
            def _at(t_win, _y=y):
                z = self._one_window(_y, phi, t_win)
                zo = self.window_profiles(_y, phi, [t_win], xi_out)[0]
                if self.mass_correction:
                    shift = (-3.0 * phi * t_win) - (self.volume_mean(z) - self.volume_mean(_y))
                    z, zo = z + shift, zo + shift
                return z, zo
            while wi < len(want) and want[wi] <= done + dtau + 1e-12:
                out.append(_at(max(want[wi] - done, 0.0))[1] + c0)
                wi += 1
            y, y_out = _at(dtau)
            done += dtau
        out.extend([y_out + c0] * (len(want) - wi))
        return y + c0, y_out + c0, out

    def _sto(self, x0, ds_mode):
        """The stoichiometry D_s(sto, T) is evaluated at for one step."""
        if ds_mode == 'ref':
            return None
        if ds_mode == 'mean':
            return self.volume_mean(x0 if len(x0) == self.n else self.resample(x0, self.n))
        if ds_mode == 'surface':
            return float(x0[-1])
        raise ValueError("ds_mode must be 'ref', 'mean' or 'surface'")

    @property
    def _vol_weights(self) -> np.ndarray:
        """Row vector v with <y> = v . y - the volume mean is linear in the sensor values."""
        if getattr(self, "_vw", None) is None:
            e = np.eye(self.n)
            self._vw = np.array([self.volume_mean(e[i]) for i in range(self.n)])
        return self._vw

    @torch.no_grad()
    def _one_window_batch(self, y0: np.ndarray, phi: np.ndarray, tau: float) -> np.ndarray:
        """Advance a BATCH of centre-pinned profiles by one window. y0: [B, n], phi: [B]."""
        b = len(phi)
        t_col = tau / self.tau_max
        x1 = torch.full((b, self.n, 1), t_col, dtype=torch.float32, device=self.device)
        x2 = torch.as_tensor(self.sensors, dtype=torch.float32, device=self.device)
        x2 = x2.view(1, self.n, 1).expand(b, self.n, 1)
        cs0 = torch.as_tensor(y0, dtype=torch.float32, device=self.device).view(b, 1, self.n)
        ph = torch.as_tensor(phi, dtype=torch.float32, device=self.device).view(b, 1, 1)
        return self.net(x1, x2, cs0, ph).squeeze(2).cpu().numpy().astype(float)

    def step_batch(self, x0: np.ndarray, I: np.ndarray, dt: float, chem: Chemistry,
                   T: float = T_REF) -> np.ndarray:
        """Advance B particles of one electrode in a single network call. x0: [B, n_sensors],
        I: [B] equivalent cell currents. Equivalent to calling `step` per row with ds_mode='ref',
        which is what makes tau - and so the sub-step schedule - common to the batch."""
        x0 = np.atleast_2d(np.asarray(x0, dtype=float))
        phi = np.array([chem.flux(float(i), T) for i in np.asarray(I, dtype=float).ravel()])
        tau = chem.fourier(float(dt), T)
        worst = float(np.max(np.abs(phi)))
        ok, msg = self.envelope_check(worst, tau)
        if not ok and not self._warned_phi:
            logger.warning("Surrogate: %s", msg)
            self._warned_phi = True

        c0 = x0[:, :1]
        y = x0 - c0                                            # centre-pinned ICs
        steps = self.substeps(float(tau))                     # same rule as step(): equal windows
        if not steps:
            return x0.copy()                                  # a zero-length step is the identity
        self.n_substeps_last = len(steps)
        v = self._vol_weights
        for dtau in steps:
            y_new = self._one_window_batch(y, phi, dtau)
            if self.mass_correction:                           # exact lithium balance per particle
                shift = (-3.0 * phi * dtau) - (y_new @ v - y @ v)
                y_new = y_new + shift[:, None]
            y = y_new
        return y + c0

    def step(self, x0: np.ndarray, I: float, dt: float, chem: Chemistry, T: float = T_REF,
             ds_mode: str = 'ref') -> np.ndarray:
        """Dimensional interface: (I [A], dt [s], T [K]) for the electrode described by `chem`.
        ds_mode: which stoichiometry the PyBaMM diffusivity D_s(sto, T) is evaluated at for THIS step -
        'ref' (chem.sto_ref, constant), 'mean' (volume mean of x0), 'surface' (x0 at r=R)."""
        x0 = np.asarray(x0, dtype=float)
        phi, tau = chem.to_dimless(I, dt, T, self._sto(x0, ds_mode))
        return self.step_dimless(x0, phi, tau)

    def __repr__(self):
        return (f"Surrogate(v={self.manifest.get('version')}, n_sensors={self.n}, "
                f"tau_max={self.tau_max}, phi_max={self.phi_max}, mass_correction={self.mass_correction}, "
                f"substeps={self.substep_policy}, "
                f"run={os.path.basename(self.run_dir)})")
