import numpy as np 
from torch.utils.data import DataLoader, TensorDataset
import torch
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass, field
from typing import List
import pybamm

def generate_grf(n, scale_length=0.2, L = 1):
    dL = L / max((n-1),1)
    SL = scale_length/dL # normalised scale length
    noise = np.random.normal(0, 1, (n,))
    grf = gaussian_filter(noise, sigma=SL)

    # Normalize the GRF
    if n > 1:
        grf -= np.mean(grf)
        grf /= np.std(grf)*2

    return grf 


HYBRID_SPLIT, HYBRID_STEP = 0.5, 0.1


def hybrid_frac(n: int, split: float = HYBRID_SPLIT, step: float = HYBRID_STEP) -> float:
    """Fraction of the hybrid budget landing in [0, split) - the node law's own CDF, so the
    collocation points and the sensors follow the SAME distribution."""
    return float(int(np.ceil((split - 1e-12) / step)) / int(n))


def xi_hybrid(s, split: float = HYBRID_SPLIT, frac: float = 0.2):
    """Map a uniform variate s on [0,1] onto the hybrid law: uniform inside `split`, equal
    VOLUME outside it. This is the inverse CDF of the node law in xi_nodes."""
    s = np.asarray(s, dtype=float)
    out = np.empty_like(s)
    lo = s < frac
    out[lo] = split * s[lo] / max(frac, 1e-12)
    u = split ** 3 + (1.0 - split ** 3) * (s[~lo] - frac) / max(1.0 - frac, 1e-12)
    out[~lo] = np.clip(u, 0.0, 1.0) ** (1.0 / 3.0)
    return out


def xi_nodes(n: int, law: str = 'uniform', p: float = 2.0,
             split: float = HYBRID_SPLIT, step: float = HYBRID_STEP):
    """n radial nodes on [0,1], ALWAYS including both ends.

    'uniform'  : equally spaced in xi.
    'particle' : equally spaced in VOLUME, xi_i = (i/(n-1))^(1/3), because a sphere's volume
                 grows as xi^3. Every node then represents the same amount of active material,
                 which is the right measure for anything integrated against xi^2 dxi - the
                 lithium inventory, the mass balance, the volume mean. It also clusters towards
                 the surface, where the boundary layer lives: at n = 41 the outermost spacing is
                 0.0084 against 0.025 uniform. The cost is a single cell spanning xi = 0 .. 0.29,
                 which makes it ~7x WORSE than uniform on a smooth profile.
    'sqrt'     : xi_i = (i/(n-1))^(1/2), density p(xi) ~ xi. The compromise between the two, and
                 the best WORST CASE of the three: measured on the volume mean over profiles with
                 boundary layers from 0.5 down to 0.02, the largest error is 2.1e-3 against
                 7.9e-3 for uniform and 4.3e-3 for particle (41 nodes).
    'warp'     : the SENSOR law, xi = 1 - (1-s)^p. Measured on the mass-balance integrand
                 3 xi^2 x over 41 nodes it is the best of the four by a wide margin - 6.3e-4 on a
                 boundary layer of 0.05 and 1.4e-7 on the gravest smooth mode, against 9.5e-4 /
                 1.8e-3 for sqrt and 3.4e-3 / 1.6e-4 for uniform. Its density diverges as
                 1/(2 sqrt(1-xi)) at the surface, which is what a trapezoid rule wants when the
                 curvature of the integrand piles up there; it is NOT a good law for a Monte-Carlo
                 loss, where a diverging density starves the interior.
    """
    s = np.linspace(0.0, 1.0, int(n))
    if law == 'uniform':
        return s
    if law == 'particle':
        return s ** (1.0 / 3.0)
    if law == 'sqrt':
        return s ** 0.5
    if law == 'warp':
        return xi_warp(s, p)
    if law == 'hybrid':
        # Uniform core + equal-volume shell. 'particle' alone resolved the boundary layer but left
        # a single cell spanning xi = 0 .. 0.368, and the measured consequence was an error blob at
        # the CENTRE worth 15 % of the profile range on the Prada anode, while that same model had
        # the best surface error of any run. This keeps the shell and puts a plain 0.1 grid under
        # it: on the real PyBaMM fields the full-profile interpolation floor falls from
        # 0.149 / 0.510 % (21 particle) to 0.014 / 0.050 % at 24 nodes, with the outermost spacing
        # essentially unchanged (0.0165 against 0.0170).
        inner = np.arange(0.0, split - 1e-12, step)
        n_out = int(n) - inner.size
        if n_out < 2:
            raise ValueError('hybrid mesh needs at least %d nodes, got %d' % (inner.size + 2, n))
        u = np.linspace(split ** 3, 1.0, n_out)
        return np.concatenate([inner, u ** (1.0 / 3.0)])
    raise ValueError("xi node law must be 'uniform', 'sqrt', 'particle', 'warp' or 'hybrid', "
                     "got %r" % (law,))


def tau_linear(s, q: float = 0.0):
    """Map a uniform variate s on [0,1] onto a LINEARLY DROPPING density in tau.

    p(u) proportional to 1 + (q-1)u, so q is the ratio of the density at tau_max to the density at
    tau=0: q=1 is uniform, q=0 drops to zero at the end of the window, q=0.25 keeps a quarter of
    the density there. Unlike the 'block' sampler this is continuous - there is no step at t_hi and
    no sub-population of points - so the early bias can be dialled smoothly instead of chosen as a
    fraction of the budget.

    Inverting F(u) = (u + b u^2) / (1 + b) with b = (q-1)/2 gives
        u = (-1 + sqrt(1 + 4 b c)) / (2 b),   c = s (1 + b)
    which reduces to u = 1 - sqrt(1-s) at q = 0 (the triangular density) and to u = s at q = 1.
    """
    s = np.asarray(s, dtype=float)
    b = (float(q) - 1.0) / 2.0
    if abs(b) < 1e-12:
        return s
    c = s * (1.0 + b)
    return (-1.0 + np.sqrt(np.maximum(1.0 + 4.0 * b * c, 0.0))) / (2.0 * b)


def xi_warp(s, p: float = 2.0):
    """Surface-clustered radial map  xi = 1 - (1 - s)^p  on the reference coordinate s in [0,1].

    p = 1 is the identity (uniform). p = 2 is the law used for the non-uniform runs: it is
    monotone, hits both ends exactly (xi(0) = 0, xi(1) = 1) and has dxi/ds = p(1-s)^(p-1) -> 0
    at the surface, so points bunch up at xi = 1 and thin out towards the centre. The induced
    density is p(xi) = 1 / (p * (1-xi)^((p-1)/p)), i.e. 1/(2*sqrt(1-xi)) for p = 2.

    The motivation is that the lithium a shell holds goes as xi^2 dxi, so the near-centre
    states carry almost none of the charge and are almost unobservable from the surface; a grid
    uniform in xi spends the same resolution on them as on the boundary layer that sets the
    surface concentration.
    """
    s = np.asarray(s, dtype=float)
    return 1.0 - (1.0 - s) ** float(p)


## Configurates the model and training process
@dataclass
class Config:
    electrode: str = "Positive"
    param_set: str = "Mohtat2020"
    b1_dim: int = 31
    b2_dim: int = 1
    hidden_dim: int = 50
    num_hidden_layers: int = 4

    Ny_pde: int = 10_000
    Ny_bc: int = 100
    Ny_ic: int = 100

    N_pde: int = 100
    N_bc: int = 100
    N_ic: int = 100

    num_batches: int = 10

    scale_length1: List[float] = field(default_factory=lambda: [0.5, 1.0])
    scale_length2: List[float] = field(default_factory=lambda: [0.1, 1.0])
    scale_factor1: float = 0.2
    scale_factor2: float = 2

    # Distribution of the dimensionless surface flux Phi.
    #   'normal'  : (default, unchanged) Phi ~ N(0, scale_factor2^2). Most of the sampling
    #               budget lands near Phi = 0, because that is where a discharge spends most
    #               of its time -- but the cold, high-rate cells the surrogate has to cover
    #               live in the tail, which is therefore thinly sampled.
    #   'uniform' : Phi ~ U(-phi_span*scale_factor2, +phi_span*scale_factor2). Same RANGE as
    #               the normal sampler out to phi_span sigma, but flat density, so the tail
    #               gets the same number of samples per unit flux as the bulk.
    phi_dist: str = 'normal'
    phi_span: float = 2.0        # half-width of the uniform interval, in units of scale_factor2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    t_tr: float = 10.0

    epochs: int = 1e3
    n_batches : int = 10
    lr: float = 3e-4
    lr_end_factor : float = 1e-2 # learning ratefactor

    loss_weights: List[float] = field(default_factory=lambda: [5, 1e3, 10])

    hard_bc: bool = False   # hard inhomogeneous Neumann ansatz at ξ=1 (surface flux)

    # Initial-profile sampler for the PDE / IC / BC sets.
    #   'grf'          : (v1, default) centre-pinned GRF of FIXED amplitude ~scale_factor1.
    #   'grf+parabola' : (v2) GRF with amplitude ~U(0, scale_factor1) PLUS the quasi-steady
    #                    profile of the previous step, -Phi_prev*xi^2/2, Phi_prev = Phi (50 %,
    #                    continuing current), independent (40 %), 0 (10 %). Covers the
    #                    deployment profiles (amplitude 0.03 .. 0.6 in x = c/c_max), which the
    #                    fixed-amplitude GRF does not (all ~0.25).
    profile_mode: str = 'grf'

    # ── Radial mesh ──────────────────────────────────────────────────────────
    # Where the branch-1 sensors sit, and how the xi collocation points are drawn.
    #   'uniform'  : xi_i = s_i = i/(b1-1)               (v1/v2 default -- unchanged)
    #   'particle' : xi_i = s_i^(1/3)                    (equal VOLUME per sensor, p(xi) ~ xi^2)
    #   'sqrt'     : xi_i = s_i^(1/2)                    (equal AREA per sensor,   p(xi) ~ xi)
    #   'surface'  : xi_i = 1 - (1 - s_i)^xi_warp_p      (clustered at the surface, p diverges)
    # A shell at xi holds lithium ~ xi^2 dxi, so on the uniform grid the innermost node stands
    # for 0.02 % of the particle contents and the outermost for 17 % -- a 4700x spread. Under
    # the p=2 warp that spread falls to 425x and 7 of 16 nodes land in xi > 0.8, where the
    # boundary layer that sets the surface concentration actually lives.
    # 'auto' = follow `geometry` below. Set it explicitly only to override that one decision.
    xi_map: str = 'auto'
    xi_warp_p: float = 2.0
    # xi law of the PDE / IC collocation points:
    #   'block' : 50 % uniform + 50 % inside xi in [0.8,1] x t in [0,0.3] (the v2 sampler)
    #   'warp'  : the same xi_warp law as the sensors, over the whole budget
    #   'uniform' / 'sqrt' / 'particle' : one Sobol draw, xi = s^(1/p) with p = 1, 2, 3
    #   'auto'  : follow `geometry`. NOTE this used to mean "'warp' if xi_map == 'surface' else
    #             'block'"; the two legacy laws are still reachable by naming them.
    xi_colloc: str = 'auto'
    # Node law for the mass-balance quadrature and, when xi_colloc='particle', for the IC
    # collocation: 'uniform' or 'particle' (equal volume per node). See xi_nodes.
    xi_quad: str = 'follow'
    # Fraction of the time column covered by the DENSE half of the PDE collocation sampler. It was
    # fixed at 0.3 when tau_max was 0.2, which put the whole deployment range (tau <= 0.047, i.e.
    # t <= 0.24) inside it. At tau_max = 0.1 deployment reaches t = 0.47 and 38 % of it falls into
    # the sparse half, which gets 0.5x uniform density against 2.17x inside. Set it to cover the
    # deployment range for whatever tau_max is in use.
    t_hi: float = 0.3
    # The GRF amplitude of the PDE and IC profiles. False (historical) multiplies every one of them
    # by exactly scale_factor1, so that ensemble has ONE amplitude; only the BC sampler draws
    # U(0, scale_factor1). With the parabola in the mix the variety came from there and the
    # asymmetry was invisible, but a pure-GRF ensemble at a fixed amplitude has no small profiles
    # at all - measured: median peak-to-peak 0.692 against 0.018 for real PyBaMM deployment
    # profiles. True draws U(0, scale_factor1) everywhere, which is what a "range up to sf" means.
    profile_amp_random: bool = False
    # Fraction of 'grf+parabola' profiles whose parabola uses the SAME Phi the second
    # branch is given. 0 keeps the shape and drops the coherence; see _profile().
    phi_prev_match: float = 0.5
    # 'shaped' profile mode (2026-10-08), see _profile. A fraction shaped_grf_frac of the profiles
    # is a pure GRF of random amplitude U(0, scale_factor1); the rest is the one-line family
    #     x0 = -Phi_prev [ (1-w) xi^2/2 + w xi^m/m ]  +  small GRF of amplitude U(0, shaped_noise)
    # with w ~ U(0,1) and m log-uniform on shaped_m_range. Every member has dx/dxi = 0 at the
    # centre and EXACTLY -Phi_prev at the surface (before the noise): m = 2 is the quasi-steady
    # parabola of a continuing current, large m a thin surface layer just after a current change.
    # Phi_prev = Phi with probability phi_prev_match, an independent N(0, sf2) draw with
    # probability shaped_indep_frac, and 0 otherwise.
    shaped_grf_frac: float = 0.2
    shaped_indep_frac: float = 0.3
    shaped_m_range: tuple = (2.0, 20.0)
    shaped_noise: float = 0.03
    # cap on the shape's depth |x0(1) - x0(0)|, enforced by redrawing (w, m) so the surface gradient
    # is untouched. 0.65 = the deepest PyBaMM profile over the whole results grid. None = no cap.
    shaped_depth_max: float = 0.65
    # (q,) linearly dropping tau density for the PDE collocation points, or None for
    # uniform. Replaces the 'block' sampler's time box rather than stacking with it.
    tau_density: tuple = None
    # Apply that same tau density to the BC set. The BC carries tau too and is where
    # -dx/dxi|_1 = Phi is enforced, i.e. the surface, where the boundary layer forms -
    # but it has always drawn tau UNIFORMLY, so only ~7 % of BC points land in the
    # deployment range. Default False so every existing config replays unchanged.
    tau_density_bc: bool = False
    # DOUBLE DOMAIN IN TIME. (frac, hi): a fraction `frac` of every collocation batch - PDE and BC
    # alike - is drawn with tau in [0, hi*tau_max] instead of [0, tau_max], the rest spanning the
    # full range. Orthogonal to the xi treatment, so the xi sampling (including the 'block'
    # surface bias) is unchanged and ONLY the time distribution moves. None = off, the historical
    # behaviour, where the only time bias is the 'block' sampler's t_hi and it applies to the
    # surface half alone.
    #
    # Why: deployment lives at small tau. The smallest-tau cells are the worst in the whole
    # scoreboard, and error against tau per call rises steeply below tau ~ 0.01 - a twentieth of
    # tau_max - which is territory a uniform tau sampler gives 5 % of its points to.
    tau_split: tuple = None
    # Where the initial-profile ENSEMBLE is drawn:
    #   'index'    : the GRF is drawn at the b1 sensor indices (legacy). The realised profiles
    #                then depend on the mesh, so a mesh change is not a single-variable change.
    #   'physical' : the GRF is drawn on a fine uniform xi grid and READ at the sensors, so the
    #                ensemble of physical profiles is identical on every mesh. Required by
    #                xi_map='surface'; use it for the uniform control too.
    profile_space: str = 'auto'
    n_fine: int = 201
    # Peak |x0(xi) - x0(0)| of the initial profile. The SHAPE keeps coming from the GRF and the
    # parabola; only the magnitude is set here, by a fresh U(0, profile_amp_max) draw.
    #
    # Without it the parabola -Phi_prev xi^2/2 inherits Phi_prev ~ N(0, PHI_SF), so with PHI_SF = 2
    # the drawn profiles had median peak 0.60 and 29 % of them exceeded 1.0. Measured over 94 944
    # snapshots of the 242 cached PyBaMM reference fields - every cell, protocol and temperature -
    # the real centre-to-surface spread has p50 = 0.031, p99 = 0.342 and a maximum of 0.649. The
    # sampler was spending most of its capacity an order of magnitude outside anything the model is
    # ever shown, and a parabola of amplitude 3 is not a stoichiometry profile at all.
    #
    # 0.5 covers 99.84 % of real snapshots and still leaves training margin. Decoupling amplitude
    # from Phi_prev is also the more physical choice: the quasi-steady parabola is only reached
    # after the flux has acted for a while, so for a given Phi_prev the real amplitude depends on
    # elapsed time - which is exactly what a free amplitude draw represents.
    # None = off (the legacy sampler).
    profile_amp_max: float = None
    # ---- surface boundary-layer shapes -------------------------------------------------------
    # A fraction of draws get their GRF multiplied by exp((xi-1)/delta), which confines the random
    # shape to a shell of thickness delta and leaves the core flat.
    #
    # Why: the parabola is the tau -> infinity quasi-steady shape, and a SCALED parabola is not a
    # young profile. Fitting the best multiple of it to the true solution under constant flux
    # leaves a residual of 80 % of the profile depth at tau = 0.002, 45 % at 0.02, and only falls
    # under 10 % past tau = 0.1. Young profiles are boundary layers - flat core, all the drop in
    # the outer shell - and nothing in a GRF of correlation length 0.5..1.0 produces that shape.
    #
    # It matters most for the cells that have always reported the worst error: at a 20 s step the
    # Prada anode accumulates tau = 0.0024 and the Chen cathode 0.0029, so they need 42 and 34
    # steps to reach tau_cum = 0.1, against ~60 steps in a 3C discharge. They spend most of a
    # protocol in exactly the regime the parabola family cannot represent.
    #
    # delta = sqrt(tau) is the diffusive layer thickness, so the deployment range tau = 0.002..0.1
    # maps to delta = 0.045..0.32. Drawn log-uniform over that.
    #
    # The LOWER bound is set by the boundary condition rather than by tau. The profile's surface
    # slope after enveloping is ~ amplitude / delta, and the hard-BC ansatz pushes that slope to
    # -Phi, so a shell thinner than profile_amp_max / (3 sigma_Phi) asks the IC term for a slope
    # the BC term will fight. With amp_max 0.5 and Phi ~ N(0,2) that floor is 0.5/6 = 0.083.
    profile_layer_frac: float = 0.0
    profile_layer_delta: tuple = (0.05, 0.35)

    # ── Geometry: ONE switch for every radial sampling decision ──────────────
    # Selecting it is enough. xi_map, xi_colloc, xi_quad and profile_space are all derived, so a
    # run cannot end up with warped sensors and uniform collocation points by forgetting a field.
    #
    #   'uniform'  : sensors, PDE/IC collocation and the mass-balance quadrature all uniform in
    #                xi. The reported configuration.
    #   'particle' : all of them non-uniform and clustered towards the surface -- where the
    #                lithium is (a shell holds ~xi^2 dxi of it), where the boundary layer that
    #                sets the surface concentration lives, and where the state is observable at
    #                all. Sensors use the p-warp 1-(1-s)^xi_warp_p; collocation and quadrature
    #                use `geometry_law`. profile_space switches to 'physical', which a warped
    #                sensor grid requires.
    #
    # Any of xi_map / xi_colloc / xi_quad / profile_space set to something other than its
    # 'auto'/'follow' sentinel overrides the preset for that one decision.
    #   'uniform'  : equally spaced in xi, everywhere.
    #   'particle' : uniform up to hybrid_split, then equally spaced in VOLUME (xi^3) out to the
    #                surface - the 'hybrid' node law.
    # There are exactly TWO, and each governs the sensors AND the PDE, IC and mass-balance points
    # together. The finer-grained laws below ('sqrt', the pure equal-volume 'particle' node law,
    # 'warp', 'block') remain reachable by setting xi_map / xi_colloc / xi_quad explicitly, which is
    # what reproducing an older run needs, but they are not geometry choices.
    geometry: str = 'uniform'
    # The collocation and quadrature law under geometry='particle'. 'sqrt' by default: over a
    # smooth profile AND a boundary layer of 0.05 its worst-case quadrature error on 41 nodes is
    # 2.1e-3, against 7.9e-3 uniform and 4.3e-3 for the equal-volume law, which leaves a single
    # cell spanning xi = 0 .. 0.29. 'particle' (s^(1/3), equal volume per point) is the matched
    # rule if the boundary layer is all that matters.
    geometry_law: str = 'sqrt'
    # The hybrid law's core: uniform up to hybrid_split in steps of hybrid_step, equal volume
    # beyond it. See xi_nodes.
    hybrid_split: float = 0.5
    hybrid_step: float = 0.1

class data():   
    
    def __init__(self, cfg = Config()):

        self.N_pde = cfg.N_pde
        self.Ny_pde = cfg.Ny_pde
        self.N_ic = cfg.N_ic
        self.Ny_ic = cfg.Ny_ic
        self.N_bc = cfg.N_bc
        self.Ny_bc = cfg.Ny_bc
        self.b1 = cfg.b1_dim
        self.b2 = cfg.b2_dim
        # self.device = cfg.device 
        self.dtype = torch.float32
        self.SL1_min = cfg.scale_length1[0]
        self.SL1_max = cfg.scale_length1[1]
        self.SL2_min = cfg.scale_length2[0]
        self.SL2_max = cfg.scale_length2[1]
        self.SF1 = cfg.scale_factor1
        self.SF2 = cfg.scale_factor2
        self.profile_mode = getattr(cfg, 'profile_mode', 'grf')
        if self.profile_mode not in ('grf', 'grf+parabola', 'shaped'):
            raise ValueError(f"profile_mode must be 'grf', 'grf+parabola' or 'shaped', "
                             f"got {self.profile_mode!r}")
        self.phi_dist = getattr(cfg, 'phi_dist', 'normal')
        if self.phi_dist not in ('normal', 'uniform'):
            raise ValueError(f"phi_dist must be 'normal' or 'uniform', got {self.phi_dist!r}")
        self.PHI_HI = float(getattr(cfg, 'phi_span', 2.0)) * self.SF2

        # ── radial mesh: one geometry switch, then per-decision overrides ────
        self.geometry = getattr(cfg, 'geometry', 'uniform')
        if self.geometry not in ('uniform', 'particle'):
            raise ValueError(f"geometry must be 'uniform' or 'particle', got {self.geometry!r}")
        self.hybrid_split = float(getattr(cfg, 'hybrid_split', 0.5))
        self.hybrid_step = float(getattr(cfg, 'hybrid_step', 0.1))
        self.hybrid_frac = hybrid_frac(int(cfg.b1_dim), self.hybrid_split, self.hybrid_step)
        # ONE law for all three roles - sensors, collocation (PDE and IC) and the mass-balance
        # quadrature. 'particle' resolves to the HYBRID node law: uniform inside hybrid_split, then
        # equal VOLUME out to the surface. Not the pure equal-volume law, which was tried and left
        # a single cell spanning xi = 0 .. 0.368; the measured cost was an error blob at the centre
        # worth 15 % of the profile range on the Prada anode.
        #
        # profile_space is 'physical' for BOTH: an index-space GRF is drawn at the sensor positions,
        # so the ensemble of physical profiles would change with the sensor count and a mesh change
        # would stop being a single-variable experiment. Reproducing a pre-2026-09-23 run needs
        # profile_space='index' set explicitly.
        _preset = (dict(xi_map='uniform', xi_colloc='uniform', profile_space='physical')
                   if self.geometry == 'uniform' else
                   dict(xi_map='hybrid', xi_colloc='hybrid', profile_space='physical'))

        self.xi_map = getattr(cfg, 'xi_map', 'auto')
        if self.xi_map == 'auto':
            self.xi_map = _preset['xi_map']
        if self.xi_map not in ('uniform', 'sqrt', 'particle', 'hybrid', 'surface'):
            raise ValueError(f"xi_map must be 'uniform', 'sqrt', 'particle', 'hybrid', 'surface' "
                             f"or 'auto', got {self.xi_map!r}")
        self.warp_p = float(getattr(cfg, 'xi_warp_p', 2.0))
        self.profile_space = getattr(cfg, 'profile_space', 'auto')
        if self.profile_space == 'auto':
            self.profile_space = _preset['profile_space']
        if self.profile_space not in ('index', 'physical'):
            raise ValueError(f"profile_space must be 'index' or 'physical', got {self.profile_space!r}")
        if self.xi_map != 'uniform' and self.profile_space != 'physical':
            # on a warped mesh an index-space GRF is a DIFFERENT ensemble of physical profiles,
            # which would confound the mesh change with a training-distribution change
            raise ValueError(f"xi_map={self.xi_map!r} requires profile_space='physical'")
        self.t_hi = float(getattr(cfg, 't_hi', 0.3))
        self.profile_amp_random = bool(getattr(cfg, 'profile_amp_random', False))
        self.phi_prev_match = float(getattr(cfg, 'phi_prev_match', 0.5))
        self.shaped_grf_frac = float(getattr(cfg, 'shaped_grf_frac', 0.2))
        self.shaped_indep_frac = float(getattr(cfg, 'shaped_indep_frac', 0.3))
        self.shaped_m_range = tuple(float(v) for v in getattr(cfg, 'shaped_m_range', (2.0, 20.0)))
        self.shaped_noise = float(getattr(cfg, 'shaped_noise', 0.03))
        _sdm = getattr(cfg, 'shaped_depth_max', 0.65)
        self.shaped_depth_max = None if _sdm is None else float(_sdm)
        if self.profile_mode == 'shaped':
            if not (0.0 <= self.shaped_grf_frac <= 1.0
                    and 0.0 <= self.phi_prev_match + self.shaped_indep_frac <= 1.0):
                raise ValueError('shaped: shaped_grf_frac and phi_prev_match + shaped_indep_frac '
                                 'must lie in [0, 1]')
            if not 1.0 < self.shaped_m_range[0] <= self.shaped_m_range[1]:
                raise ValueError(f'shaped_m_range must be 1 < lo <= hi (m > 1 keeps the centre '
                                 f'gradient zero), got {self.shaped_m_range!r}')
        self.tau_density = getattr(cfg, 'tau_density', None)
        self.tau_density_bc = bool(getattr(cfg, 'tau_density_bc', False))
        ts = getattr(cfg, 'tau_split', None)
        if ts is not None:
            ts = (float(ts[0]), float(ts[1]))
            if not (0.0 < ts[0] < 1.0) or not (0.0 < ts[1] <= 1.0):
                raise ValueError(f"tau_split must be (frac in (0,1), hi in (0,1]), got {ts}")
        self.tau_split = ts
        self.xi_quad = getattr(cfg, 'xi_quad', 'follow')
        if self.xi_quad not in ('uniform', 'sqrt', 'particle', 'warp', 'hybrid', 'follow'):
            raise ValueError(f"xi_quad must be 'uniform', 'sqrt', 'particle', 'warp', 'hybrid' "
                             f"or 'follow', got {self.xi_quad!r}")
        self.xi_colloc = getattr(cfg, 'xi_colloc', 'auto')
        if self.xi_colloc == 'auto':
            self.xi_colloc = _preset['xi_colloc']
        if self.xi_colloc not in ('uniform', 'sqrt', 'particle', 'hybrid', 'block', 'warp',
                                 'tblock'):
            raise ValueError(f"xi_colloc must be 'uniform', 'sqrt', 'particle', 'hybrid', "
                             f"'block', 'tblock', 'warp' or 'auto', got {self.xi_colloc!r}")
        # The mass-balance quadrature follows the same law unless overridden, so one setting
        # governs every radial sampling decision in the run.
        if getattr(cfg, 'xi_quad', None) in (None, 'follow'):
            self.xi_quad = (self.xi_colloc
                            if self.xi_colloc in ('uniform', 'sqrt', 'particle', 'hybrid', 'warp')
                            else 'uniform')

        s_ref = np.linspace(0, 1, self.b1)                      # reference coordinate
        self.sensors1 = (xi_warp(s_ref, self.warp_p) if self.xi_map == 'surface'
                         else xi_nodes(self.b1, self.xi_map, split=self.hybrid_split,
                                       step=self.hybrid_step))
        self.sensors2 = np.linspace(0,1,self.b2) #Position of sensors
        self.layer_frac = float(getattr(cfg, 'profile_layer_frac', 0.0) or 0.0)
        _ld = getattr(cfg, 'profile_layer_delta', (0.05, 0.35))
        self.layer_delta = (float(_ld[0]), float(_ld[1]))
        if not 0.0 <= self.layer_frac <= 1.0:
            raise ValueError(f"profile_layer_frac must be in [0,1], got {self.layer_frac!r}")
        if not 0 < self.layer_delta[0] <= self.layer_delta[1]:
            raise ValueError(f"profile_layer_delta must be 0 < lo <= hi, got {self.layer_delta!r}")
        self.profile_amp_max = getattr(cfg, 'profile_amp_max', None)
        if self.profile_amp_max is not None and float(self.profile_amp_max) <= 0:
            raise ValueError(f"profile_amp_max must be positive or None, got {self.profile_amp_max!r}")
        self.n_fine = int(getattr(cfg, 'n_fine', 201))
        self.xi_fine = np.linspace(0.0, 1.0, self.n_fine)
        self.num_batches = cfg.num_batches
        
        try:
            param = pybamm.ParameterValues(cfg.param_set)
        except Exception as e:
            raise RuntimeError(
                f"Could not load PyBaMM parameter values from '{cfg.param_set}'"
            ) from e
        
        self.param = param
        self.t_tr = cfg.t_tr

        self.C = param['Nominal cell capacity [A.h]']

        if cfg.electrode == 'Negative' or cfg.electrode == 'Positive':
            self.Electrode = cfg.electrode
        else: 
            raise ValueError('Electrode must be either "Negative" or "Positive".')
        
        tau = 0.1 # --> x>2*tau, g(x) = 1
        # g is a function of the PHYSICAL coordinate xi (on the uniform mesh sensors1 = s, so
        # this is exactly the old np.linspace(0, 1/tau, b1); on a warped mesh it keeps the
        # centre taper anchored to xi rather than to the sensor index).
        self.g = 1 - np.exp(-(self.sensors1/tau)**2); ##we force to be f1 at t = 0, bu we also force the concentration to be zero and have a null derivative at the center of the domain
        # self.g = np.ones(self.b1,)

        # Soft surface window: ~1 only in the outer shell near ξ=1, decaying to ~0
        # in the bulk (mirror of g, which tames the centre). Used by
        # _soften_surface to damp steep/noisy GRF surfaces WITHOUT pinning the
        # surface value or slope to any current (cf. the chosen "soft window").
        x_surf = (1.0 - self.sensors1) / tau            # 0 at ξ=1, large in the bulk
        self.g_surf = np.exp(-x_surf**2)                 # 1 at surface, →0 for ξ ≲ 0.8
        # the same two windows on the fine grid used by profile_space='physical'
        self.g_fine = 1 - np.exp(-(self.xi_fine/tau)**2)
        self.g_surf_fine = np.exp(-((1.0 - self.xi_fine)/tau)**2)
        # _soften_surface smooths with sigma = 2 SAMPLES of the b1 grid, i.e. 2/(b1-1) in xi.
        # On the fine grid the same physical length is 2*(n_fine-1)/(b1-1) samples, so the
        # 'physical' path softens by the same amount of xi as the legacy path.
        self._soft_sigma_fine = 2.0 * (self.n_fine - 1) / max(self.b1 - 1, 1)

    def _draw_phi(self, scl, amp_scale=None):
        """The second branch input: the dimensionless surface flux, as an array of length b2.

        'normal'  reproduces the original sampler exactly -- a unit GRF scaled by SF2 (or, for
                  the boundary set, by a random amplitude in [0, SF2]).
        'uniform' draws a scalar flat on [-PHI_HI, +PHI_HI] and broadcasts it. b2 = 1 in the
                  dimensionless formulation (the flux is one number per step), so broadcasting
                  is the whole field; for b2 > 1 each sensor is drawn independently, which keeps
                  the marginal uniform.
        """
        if self.phi_dist == 'uniform':
            return np.random.uniform(-self.PHI_HI, self.PHI_HI, size=self.b2)
        s = self.SF2 if amp_scale is None else amp_scale
        return generate_grf(self.b2, scale_length=scl) * s

    def _draw_phi_const(self):
        """The every-tenth-profile constant-flux branch. In 'normal' mode this reproduces the
        original `np.ones_like(f2) * np.random.normal(0, 1)`, which is NOT scaled by SF2 -- a
        quirk of the original sampler, preserved so that the default path is unchanged. In
        'uniform' mode it is drawn from the same interval as everything else, so the whole
        flux distribution is flat rather than flat with a narrow Gaussian spike at the centre."""
        if self.phi_dist == 'uniform':
            return np.full(self.b2, np.random.uniform(-self.PHI_HI, self.PHI_HI))
        return np.ones(self.b2) * np.random.normal(0, 1)

    def _amp(self):
        """GRF amplitude for the PDE and IC ensembles: fixed at scale_factor1 historically,
        U(0, scale_factor1) when profile_amp_random is on (as the BC sampler has always done)."""
        return np.random.uniform(0, self.SF1) if self.profile_amp_random else None

    def _profile(self, phi, amp=None, return_parts=False):
        """One centre-pinned initial profile x0(xi) at the sensors (see Config.profile_mode).
        `phi`: the flux of THIS step (scalar); `amp`: GRF amplitude (default scale_factor1).

        With profile_space='physical' the profile is built on the fine uniform xi grid and only
        READ at the sensors, so the ensemble of physical profiles does not depend on where the
        sensors are. generate_grf normalises its scale_length by the grid spacing, so the
        correlation length is the same in xi on either grid; the draw order of the random
        numbers is identical, so the 'index' path is unchanged."""
        scl = np.random.uniform(self.SL1_min, self.SL1_max)
        a = self.SF1 if amp is None else amp
        if self.profile_space == 'physical':
            grid, g, g_surf, sig = self.xi_fine, self.g_fine, self.g_surf_fine, self._soft_sigma_fine
            f1 = generate_grf(self.n_fine, scale_length=scl) * a
        else:
            grid, g, g_surf, sig = self.sensors1, self.g, self.g_surf, 2.0
            f1 = generate_grf(self.b1, scale_length=scl) * a
        f1 = f1 - f1[0]                   # concentration in the centre is always null (observer convention)
        f1 = f1 * g
        f1 = self._soften_surface(f1, g_surf, sig)    # tame steep/noisy GRF surface slopes
        parts = dict(scale_length=scl, grf_scale=1.0, phi_prev=0.0, branch='grf')
        if self.profile_mode == 'shaped':
            if np.random.uniform() < self.shaped_grf_frac:
                # pure GRF, at a RANDOM amplitude so that small profiles exist in this branch too
                gs = np.random.uniform(0.0, 1.0)
                f1 = f1 * gs
                parts.update(grf_scale=gs)
            else:
                u = np.random.uniform()
                mt, it = self.phi_prev_match, self.shaped_indep_frac
                phi_prev = (float(phi) if u < mt else
                            np.random.normal(0.0, 1.0) * self.SF2 if u < mt + it else 0.0)
                lo, hi = np.log(self.shaped_m_range[0]), np.log(self.shaped_m_range[1])
                # depth = |Phi_prev| [(1-w)/2 + w/m]. Redraw (w, m) while it exceeds shaped_depth_max:
                # a large flux then lands on a sharp, shallow surface layer - the profile right after
                # a big current step - instead of a parabola deeper than any real profile, and the
                # surface gradient stays exactly -Phi_prev. Gives up after 50 draws (only possible
                # for |Phi_prev| > shaped_depth_max * m_max, far outside the sampled flux).
                for _ in range(50):
                    w = np.random.uniform(0.0, 1.0)
                    m = float(np.exp(np.random.uniform(lo, hi)))
                    if (self.shaped_depth_max is None or
                            abs(phi_prev) * ((1.0 - w) / 2.0 + w / m) <= self.shaped_depth_max):
                        break
                # surface gradient exactly -Phi_prev for every (w, m); centre gradient 0 for m > 1
                shape = -phi_prev * ((1.0 - w) * grid ** 2 / 2.0 + w * grid ** m / m)
                gs = np.random.uniform(0.0, self.shaped_noise) / max(a, 1e-12)
                f1 = shape + f1 * gs                    # the GRF is now only a perturbation
                parts.update(grf_scale=gs, phi_prev=phi_prev, w=w, m=m,
                             branch=('matched' if u < mt else 'independent' if u < mt + it
                                     else 'relaxed'))
        if self.profile_mode == 'grf+parabola':
            # The GRF amplitude is scaled by U(0,1) rather than used at full strength, so the set
            # spans everything from a pure parabola (scale ~ 0) to a fully developed random shape.
            gs = np.random.uniform(0.0, 1.0)
            f1 = f1 * gs
            # ...and the parabola comes from one of three branches, which is what puts both
            # matched and mismatched (x0, Phi) pairs in the set. Without the mismatched half the
            # network would only ever see profiles consistent with the flux it is given.
            # phi_prev_match is the MATCHED fraction; the zero branch keeps its 10 % and the
            # independent branch takes the rest. Setting it to 0 keeps the parabolic SHAPE but
            # destroys the consistency between x0 and the flux the second branch is given, which
            # is the only way to tell "realistic shape" apart from "coherent (x0, Phi) pair".
            m = float(getattr(self, 'phi_prev_match', 0.5))
            u = np.random.uniform()
            phi_prev = (float(phi) if u < m else
                        np.random.normal(0.0, 1.0) * self.SF2 if u < 0.9 else 0.0)
            parts.update(grf_scale=gs, phi_prev=phi_prev,
                         branch=('matched' if u < m else 'independent' if u < 0.9 else 'grf'))
            f1 = f1 - phi_prev * grid ** 2 / 2.0               # quasi-steady profile of the previous step
            # Confine the WHOLE profile to a surface shell. It has to come after the parabola, not
            # be applied to the GRF alone: the parabola carries ~4x the GRF amplitude, so an
            # enveloped GRF is simply swamped by it and the assembled shape stays parabola-like
            # (measured: the fraction of boundary-layer shapes moved 23.0 % -> 24.1 %, i.e. not at
            # all). Enveloping the assembled profile is what actually produces a flat core with the
            # whole drop in the outer shell.
            if self.layer_frac > 0.0 and np.random.uniform() < self.layer_frac:
                lo, hi = np.log(self.layer_delta[0]), np.log(self.layer_delta[1])
                dlt = float(np.exp(np.random.uniform(lo, hi)))
                f1 = f1 * np.exp((grid - 1.0) / dlt)
                parts['layer_delta'] = dlt
        if self.profile_amp_max is not None:
            # Shape is already decided; this only sets the magnitude. See Config.profile_amp_max.
            peak = float(np.abs(f1).max())
            amp_t = np.random.uniform(0.0, float(self.profile_amp_max))
            parts['amp_target'] = amp_t
            if peak > 1e-12:
                f1 = f1 * (amp_t / peak)
        if self.profile_space == 'physical':
            parts['fine'] = f1.copy()
            f1 = np.interp(self.sensors1, self.xi_fine, f1)    # read the profile at the sensors
        return (f1, parts) if return_parts else f1

    def _soften_surface(self, f1, g_surf=None, sigma=2.0):
        """Blend the near-surface region of an IC profile toward an extra-smoothed
        copy so a handful of steep/noisy GRF surface slopes can't produce
        unphysical profiles. The bulk is untouched; the surface value/slope are
        left free (no coupling to current or j0_aux). `sigma` is in SAMPLES of
        whichever grid f1 lives on."""
        g_surf = self.g_surf if g_surf is None else g_surf
        f1_s = gaussian_filter(f1, sigma=sigma, mode="nearest")
        return (1.0 - g_surf) * f1 + g_surf * f1_s


    def _tau_split(self, pts):
        """Compress the time column of a random `frac` of `pts` into [0, hi].

        Applied AFTER the xi treatment and to the concatenated budget, so it cuts across both
        halves of the block sampler: the xi distribution is untouched and exactly `frac` of the
        points - surface-biased and bulk alike - end up in the early-time domain. The draw is a
        permutation rather than a slice because the Sobol rows are ordered, and taking a prefix
        would correlate the time domain with the sequence's own stratification.
        """
        if self.tau_split is None or len(pts) == 0:
            return pts
        frac, hi = self.tau_split
        idx = np.random.permutation(len(pts))[:int(round(frac * len(pts)))]
        pts[idx, 0] = pts[idx, 0] * hi
        return pts

    def data_pde(self, N_pde=None, Ny_pde=None):
        
        if N_pde == None or Ny_pde == None :
            N_pde = self.N_pde
            Ny_pde = self.Ny_pde
                
        X_aux = np.zeros([1, Ny_pde, 2 + self.b1 + self.b2])

        # ── Surface-biased collocation ────────────────────────────────────────
        # At small Fourier number the action is a thin boundary layer near ξ=1 and
        # the early transient near t=0; uniform Sobol wastes ~85 % of points in the
        # inert bulk. Spend SURF_FRAC of the budget inside ξ∈[ξ_lo,1] × t∈[0,t_hi]
        # and the rest uniformly so the bulk PDE is still constrained.
        SURF_FRAC = 0.5
        XI_LO, T_HI = 0.8, float(getattr(self, 't_hi', 0.3))

        n_surf = int(round(SURF_FRAC * Ny_pde))
        n_unif = Ny_pde - n_surf

        soboleng_pde = torch.quasirandom.SobolEngine(dimension=2)
        pts_unif = soboleng_pde.draw(n_unif).detach().numpy()          # [t, ξ] in [0,1]²
        pts_surf = soboleng_pde.draw(n_surf).detach().numpy()
        # A linear tau density OWNS the time axis: it is the smooth alternative to the step at
        # t_hi, so the time box is switched off while it is on, exactly as tau_split does.
        _tau_dens = getattr(self, 'tau_density', None)
        if self.tau_split is None and _tau_dens is None:
            pts_surf[:, 0] = pts_surf[:, 0] * T_HI                      # t  → [0, t_hi]
        if self.xi_colloc in ('uniform', 'sqrt', 'particle', 'hybrid', 'tblock'):
            # ONE Sobol draw over the whole budget. tau stays uniform - the network maps
            # (x0, Phi, tau) -> x(tau) directly, so every tau is an independent query and there is
            # no causal chain that would justify weighting early times. xi follows the law:
            #   'uniform'  : as drawn
            #   'particle' : xi = s^(1/3), so the density is p(xi) ~ xi^2 and each point carries
            #                the same amount of active material.
            # 'tblock' is the one exception: it KEEPS the time block above and leaves xi uniform,
            # so half the budget sits in tau in [0, t_hi] across the whole radius. 'block' applies
            # both boxes to the same half of the budget and there is no way to tell from it which
            # of the two is doing the work; 'tblock' against 'block' with t_hi=1 separates them.
            if self.tau_split is None and self.xi_colloc != 'tblock':
                pts_surf[:, 0] = pts_surf[:, 0] / max(T_HI, 1e-12)    # undo the time block
            if self.xi_colloc == 'hybrid':
                pts_unif[:, 1] = xi_hybrid(pts_unif[:, 1], self.hybrid_split, self.hybrid_frac)
                pts_surf[:, 1] = xi_hybrid(pts_surf[:, 1], self.hybrid_split, self.hybrid_frac)
                _p = 1.0
            else:
                _p = {'uniform': 1.0, 'tblock': 1.0,
                      'sqrt': 0.5, 'particle': 1.0 / 3.0}[self.xi_colloc]
            if _p != 1.0:
                pts_unif[:, 1] = pts_unif[:, 1] ** _p
                pts_surf[:, 1] = pts_surf[:, 1] ** _p
        elif self.xi_colloc == 'warp':
            # ONE smooth law over the whole budget instead of the two-block step: the ξ density
            # becomes 1/(p (1-ξ)^((p-1)/p)) -- continuous, and unlike the block sampler it keeps
            # refining inside the last per-cent of the radius, where the Neumann condition acts.
            # The TIME split above is left exactly as it is, so only the ξ measure changes.
            pts_unif[:, 1] = xi_warp(pts_unif[:, 1], self.warp_p)
            pts_surf[:, 1] = xi_warp(pts_surf[:, 1], self.warp_p)
        else:
            pts_surf[:, 1] = XI_LO + pts_surf[:, 1] * (1.0 - XI_LO)     # ξ  → [ξ_lo, 1]

        coll_points_pde = np.concatenate([pts_unif, pts_surf], axis=0)
        if _tau_dens is not None:
            coll_points_pde[:, 0] = tau_linear(coll_points_pde[:, 0], float(_tau_dens[0]))
        coll_points_pde = self._tau_split(coll_points_pde)

        # The PDE residual has a 2·y_ξ/ξ term that is singular at ξ=0; clamp the
        # whole near-center shell (not just the exact-zero Sobol point) to ξ=eps
        # so a handful of O(1/ξ) residuals can't dominate the PDE loss.
        eps = 0.01
        coll_points_pde[coll_points_pde[:, 1] < eps, 1] = eps

        first_time = True


        
        for idx in range(N_pde):
            
            # f1 = F1[idx,:]
            # f2 = F2[idx,:]
            scl = np.random.uniform(self.SL2_min, self.SL2_max)
            f2 = self._draw_phi(scl)   # Current is normalized --> is multiplied later

            if idx % 10 == 0:
                f2 = self._draw_phi_const()

            f1 = self._profile(f2[0], amp=self._amp())   # centre-pinned GRF (+ parabola in v2 mode)
            
            if idx % 10 == 1:
                f1 = np.zeros_like(f1)

            X_aux[0,:,:2]=coll_points_pde
            X_aux[0,:,2:2+self.b1]= f1
            X_aux[0,:,-self.b2:]= f2       
            
            if first_time:
                
                X_pde = X_aux.copy()
                first_time = False
            
            else:
                X_pde = np.concatenate((X_pde,X_aux),axis = 0)
    
        
        return torch.tensor(X_pde, dtype=self.dtype) #, device=self.device) 
 
    def data_ic(self, N_ic=None, Ny_ic=None):
        
        if N_ic == None or Ny_ic == None :
            N_ic = self.N_ic
            Ny_ic = self.Ny_ic
            
        X_aux = np.zeros([1, Ny_ic, 2 + self.b1 + self.b2 + 1])
        
        soboleng_ic = torch.quasirandom.SobolEngine(dimension=1)
        coll_points_ic =  soboleng_ic.draw(Ny_ic).detach().numpy()
        if self.xi_colloc in ('uniform', 'sqrt', 'particle', 'hybrid',
                              'tblock'):                               # same law as the PDE set
            if self.xi_colloc == 'hybrid':
                coll_points_ic = xi_hybrid(coll_points_ic, self.hybrid_split, self.hybrid_frac)
            else:
                coll_points_ic = coll_points_ic ** {'uniform': 1.0, 'tblock': 1.0, 'sqrt': 0.5,
                                                    'particle': 1.0 / 3.0}[self.xi_colloc]
        elif False:                                 # same ξ measure as the PDE set
            coll_points_ic = coll_points_ic ** (1.0 / 3.0)
        elif self.xi_colloc == 'warp':
            coll_points_ic = xi_warp(coll_points_ic, self.warp_p)
        # The IC has to be pinned at BOTH ends: xi=0 fixes the centre value the whole profile is
        # measured against, xi=1 is where the surface concentration is read out. A quasi-random
        # sample contains neither, so they are forced in.
        coll_points_ic[0, 0], coll_points_ic[-1, 0] = 0.0, 1.0
        
        first_time = True
        
        for idx in range(N_ic):
            
            # f1 = F1[idx,:]
            # f2 = F2[idx,:]
            
            scl = np.random.uniform(self.SL2_min, self.SL2_max)
            f2 = self._draw_phi(scl)   # Current is normalized --> is multiplied later

            ## Every 10 profiles, impose also a constant one
            if idx % 10 == 2:
                f2 = self._draw_phi_const()
            f1 = self._profile(f2[0], amp=self._amp())   # centre-pinned GRF (+ parabola in v2 mode)
            if idx % 10 == 3:
                f1 *= 0

            f1x = np.interp(coll_points_ic, self.sensors1, f1) # Function evaluated in the coordinate
            
            X_aux[0,:,1:2]=coll_points_ic
            X_aux[0,:,2:2+self.b1] = f1
            X_aux[0,:,2+self.b1:2+self.b1+self.b2] = f2
            X_aux[0,:,-1:] = f1x
            
            if first_time:
                X_ic = X_aux.copy()
                first_time = False
            else:
                X_ic = np.concatenate((X_ic,X_aux),axis = 0)
    
        return torch.tensor(X_ic, dtype=self.dtype) #, device=self.device) 
    
    def data_bc(self, N_bc=None, Ny_bc=None):
        
        if N_bc == None or Ny_bc == None :
            N_bc = self.N_bc
            Ny_bc = self.Ny_bc
        
        X_aux = np.zeros([1, Ny_bc, 2 + self.b1 + self.b2 + 1])
        
        soboleng_bc = torch.quasirandom.SobolEngine(dimension=1)
        coll_points_bc =  soboleng_bc.draw(Ny_bc).detach().numpy()
        if self.tau_density is not None and self.tau_density_bc:
            coll_points_bc = tau_linear(coll_points_bc, float(self.tau_density[0]))
        coll_points_bc = self._tau_split(coll_points_bc)   # BC carries tau too, same double domain

        first_time = True
        
        for idx in range(N_bc):
           
            scl = np.random.uniform(self.SL2_min, self.SL2_max)
            f2 = self._draw_phi(scl, amp_scale=np.random.uniform(0, self.SF2))   # Current is normalized --> is multiplied later

            ## Every 10 profiles, impose also a constant one
            if idx % 10 == 2:
                f2 = self._draw_phi_const()
            f1 = self._profile(f2[0], amp=np.random.uniform(0, self.SF1))   # (+ previous-step parabola in v2 mode)
            if idx % 10 == 3:
                f1 *= 0

            f2x = np.interp(coll_points_bc, self.sensors2, f2) # Function evaluated in the coordinate
            
            X_aux[0,:,:1]=coll_points_bc
            X_aux[0,:,1] = 1
            X_aux[0,:,2:2+self.b1] = f1
            X_aux[0,:,2+self.b1:2+self.b1+self.b2] = f2
            X_aux[0,:,-1:] = f2x
            
            if first_time:
                
                X_bc = X_aux.copy()
                first_time = False
            
            else:
                X_bc = np.concatenate((X_bc,X_aux),axis = 0)
    
        return torch.tensor(X_bc, dtype=self.dtype) #, device=self.device) 
    
    def data_val(self):
        """
        Idea is to use reference data for a 'DD' validation
        """
        import pybamm
        
        T_sim = 20000; dT = 1
        res = self.b1
        T_tr = self.t_tr*1.2 # Time used for training the nn model

        batt_model = pybamm.lithium_ion.SPM()
        # param = batt_model.default_parameter_values
        self.param.set_initial_stoichiometries(0.5)
        
        
        time_w = 600

        T_end = 0
        while T_end < 0.8*T_sim:

            t_eval = np.linspace(0, T_sim , int(T_sim/dT)+1)
            current = np.ones([len(t_eval)])

            k = 0
            for t in t_eval:
                if t%time_w==0:
                    II = np.clip(np.random.normal(0, 0.7), -1.5, 1.5)  # limit to ±1.5 C-rate
                current[k]=II
                k += 1

            current*= self.C

            var_pts = {
            "x_n": 10,  # negative electrode
            "x_s": 10,  # separator
            "x_p": 10,  # positive electrode
            "r_n": res,  # negative particle
            "r_p": res,  # positive particle
            }

            # Defining the experiment
            drive_cycle = np.column_stack([t_eval, current])
            experiment = pybamm.Experiment([pybamm.step.current(drive_cycle)])

            sim = pybamm.Simulation(batt_model, parameter_values=self.param, var_pts=var_pts, experiment = experiment)
            try:
                solution = sim.solve()
            except pybamm.SolverError:
                print("Solver failed (NaNs), regenerating current profile...")
                continue

            ## Results
            time1 = solution["Time [s]"].entries
            T_end = time1[-1]
            print("Pybam solved up to: ", time1[-1])

        TT = time1[-1]
        t_eval = t_eval[t_eval < TT]
        time1 = t_eval
        y_true =  solution[f"X-averaged {self.Electrode.lower()} particle concentration"](t_eval)
        current = solution["Current [A]"](t_eval)

        print('Nan in y_true:', np.isnan(y_true).any())
        print('Nan in current:', np.isnan(current).any())
        print('Max current:', np.max(current))
        print('Min current:', np.min(current))
        print('Dt max:', np.max(np.diff(time1)))
        print('Dt min:', np.min(np.diff(time1)))
        # c0 = y_true[0,0]
        # y_true -= c0

        val_pts = res
        
        X_dd = np.zeros([1,res*len(time1),2+self.b1+self.b2+1])
        r = np.linspace(0,1,res)
        k = 0

        for j in range(len(time1)):
            
            Time = time1[j]%(T_tr)
            
            if Time == 0:
                
                c0 = y_true[0,j].copy()
                cs_0 = np.interp(self.sensors1,r,y_true[:,j] - c0)[None,:]
                k+= 1
                
            X_dd[0,val_pts*j:val_pts*(j+1),0] = Time/self.t_tr # Normalize the time
            X_dd[0,val_pts*j:val_pts*(j+1),1] = r[-val_pts:]
            X_dd[0,val_pts*j:val_pts*(j+1),2:2+self.b1] = cs_0
            X_dd[0,val_pts*j:val_pts*(j+1),2+self.b1:2+self.b1+self.b2] = current[j]/self.C # Normalize the current
            X_dd[0,val_pts*j:val_pts*(j+1),-1] = y_true[-val_pts:,j] - c0

        return torch.tensor(X_dd, dtype = self.dtype) #, device = self.device) 
    
    def data_val_pi(self, n_pde=50, ny_pde=1000, n_ic=200, ny_ic=100, n_bc=200, ny_bc=100):
        """A held-out PI set: fresh profiles, fluxes and collocation points from the same
        distribution as the training sets, never trained on. Sized independently because it is
        evaluated every epoch - a validation estimate needs enough points to be tight, not as many
        as training."""
        x_val_pde = self.data_pde(n_pde, ny_pde)
        x_val_ic = self.data_ic(n_ic, ny_ic)
        x_val_bc = self.data_bc(n_bc, ny_bc)

        return (x_val_pde, x_val_ic, x_val_bc)

    def prepare_data_pi(self):
        """Physics-only variant: skips the PyBaMM simulation entirely.
        Use when w_dd=0 so the dummy val loader is never touched by the training loop."""
        X_pde = self.data_pde()
        X_ic  = self.data_ic()
        X_bc  = self.data_bc()
        loader_pde = create_loader(X_pde, self.num_batches)
        loader_ic  = create_loader(X_ic,  self.num_batches)
        loader_bc  = create_loader(X_bc,  self.num_batches)
        dummy_shape = (1, 1, 2 + self.b1 + self.b2 + 1)
        dummy_val = [torch.zeros(dummy_shape, dtype=self.dtype)] * self.num_batches
        return (loader_pde, loader_ic, loader_bc, dummy_val)

    def prepare_data(self):
        X_pde = self.data_pde()
        X_ic = self.data_ic()
        X_bc = self.data_bc()
        X_val = self.data_val() # Validation data are prepared in a slgithly different manner

        print(X_val.shape)

        t_val_max = max(X_val[:1,:,0])
        print(max(t_val_max))
        # Create DataLoaders
        num_batches = self.num_batches
        loader_pde = create_loader(X_pde, self.num_batches)
        loader_ic = create_loader(X_ic, self.num_batches)
        loader_bc = create_loader(X_bc, self.num_batches)
        loader_val = np.array_split(X_val, self.num_batches, axis=1)

        return (loader_pde, loader_ic, loader_bc, loader_val)

def create_loader(tensor, num_batches, shuffle=True):
    dataset_size = len(tensor)
    batch_size = dataset_size // num_batches

    # To handle cases where the dataset isn't perfectly divisible
    if dataset_size % num_batches != 0:
        batch_size += 1

    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=batch_size,
        shuffle=shuffle
    )
    return loader

if __name__ == "__main__":

    Data = data()
    x_dpe = Data.data_pde()

    # X_dd = Data.data_val().cpu().numpy()

    # import matplotlib.pyplot as plt
    # idx = 540
    # plt.figure()
    # plt.plot(Data.sensors1, X_dd[0,idx,2:2+Data.b1], 'orange')
    # plt.plot(X_dd[0,idx:idx+100,1],X_dd[0,idx:idx+100,-1], 'o')
    # plt.show()

    # # loaders = Data.prepare_data()
    # print('Done')

    n = 101
    x = np.linspace(0,1,n)
    f_x = generate_grf(n, scale_length=1)

    import matplotlib.pyplot as plt 

    plt.plot(x, f_x)
    plt.show()
    