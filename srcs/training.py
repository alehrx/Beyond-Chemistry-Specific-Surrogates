from dataclasses import dataclass
import torch 
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset
import math
from torch.optim.lr_scheduler import LinearLR
from typing import Dict, List, Optional, Tuple
import pybamm

from .prepData import xi_warp

## Configurates the model and training process
@dataclass
class TrainerConfig:
    electrode: str = "Negtative"
    param_set: str = "Mohtat2020"
    b1_dim: int = 50
    b2_dim: int = 50
    t_tr: float = 100.0
    epochs: int = 1e3
    lr: float = 3e-4
    lr_end_factor : float = 1e-2 # learning ratefactor
    save_every_epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 25# log every n epochs
    rollout_K: int = 0         # recursive PI rollout steps (0 = disabled)
    w_rollout: float = 1.0     # weight for rollout loss term

class Trainer():
    def __init__(self, model: nn.Module, cfg = TrainerConfig, tracker = None):
        """
        T_tr is the training time !!!
        """
        self.network = model.to(cfg.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=cfg.lr)
        self.device = cfg.device
        self.b1 = cfg.b1_dim # Update 
        self.b2 = cfg.b2_dim # Update 
        # Sensor positions in xi. Must match srcs.prepData.data.sensors1 -- the branch-1 input
        # is a vector of profile VALUES, so the network's notion of where those values sit is
        # fixed entirely by the sampler. Anything that interpolates the state (the rollout IC
        # target, the rollout advance step) has to use the same grid.
        _s = np.linspace(0, 1, self.b1)
        self.sensors1 = (_s if getattr(cfg, 'xi_map', 'uniform') == 'uniform'
                         else xi_warp(_s, getattr(cfg, 'xi_warp_p', 2.0)))
        self.T_tr = cfg.t_tr
        Electrode = cfg.electrode
        self.num_epochs = cfg.epochs
        self.scheduler = LinearLR(self.optimizer,
                        start_factor=1.0,  # start at 100% of 0.01
                        end_factor=cfg.lr_end_factor,    # end at 0 (zero learning rate)
                        total_iters=self.num_epochs    
                        )
        self.W_pde = cfg.loss_weights[0]
        self.W_ic = cfg.loss_weights[1]
        self.W_bc = cfg.loss_weights[2]
        # 4th weight (optional): mass-conservation term. Absent -> 0 -> disabled,
        # so every existing 3-element loss_weights config behaves exactly as before.
        self.W_cons = float(cfg.loss_weights[3]) if len(cfg.loss_weights) > 3 else 0.0
        self.rollout_K      = getattr(cfg, "rollout_K", 0)
        self.w_rollout      = getattr(cfg, "w_rollout", 1.0)
        self.w_dd           = getattr(cfg, "w_dd", 0.0)
        # Max collocation points fed to loss_pi_rollout per step.
        # Full training batches (Ny_pde=10000, K=20, create_graph=True) exhaust
        # GPU cuBLAS workspace. 300 pts per step is enough to detect recursive drift.
        self.rollout_n_pts  = getattr(cfg, "rollout_n_pts", 300)
        self.tracker = tracker
        self.change_opt = True

        ## EM parameter values
        try:
            param = pybamm.ParameterValues(cfg.param_set)
        except Exception as e:
            raise RuntimeError(
                f"Could not load PyBaMM parameter values from '{cfg.param_set}'"
            ) from e
        
        self.param = param
        
        ## Some of the PI loss terms depend on some physical values 
        Diff = param[f'{Electrode} electrode diffusivity [m2.s-1]']
        self.Ds=Diff(0.5,298.15).value

        F = param['Faraday constant [C.mol-1]'] 
        epsilon = param[f'{Electrode.capitalize()} electrode active material volume fraction']
        self.Rs = param[f'{Electrode.capitalize()} particle radius [m]']
        L = param[f'{Electrode.capitalize()} electrode thickness [m]']
        cs_max = param[f'Maximum concentration in {Electrode.lower()} electrode [mol.m-3]']
        # Automatically assigns parameter values based on the selected electrode
        a = 3*epsilon/self.Rs
        W=param['Electrode width [m]'] # Electrode height [m]
        H=param['Electrode height [m]']
        n_el = param['Number of electrodes connected in parallel to make a cell']
        self.C = param['Nominal cell capacity [A.h]']
        self.j0_aux = n_el*W*H*self.Ds*cs_max*F*a*L/self.Rs ## Appears in BC2

        if Electrode == 'Negative' or Electrode == 'Positive':
            self.sign = 1 if Electrode == 'Negative' else -1   ### dc/dksi(ksi = 1) = -i*sign*C/j0_aux
        else:
            raise ValueError('Electrode must be either "Negative" or "Positive".')

        # Hard inhomogeneous Neumann ansatz at ξ=1 (carries the surface flux
        # exactly; loss_bc2 then only enforces the homogeneous remainder). All
        # losses call self.network(...) so they pick this up automatically; the
        # constants here match utils.compute_bc2_constants used at deployment.
        if getattr(cfg, "hard_bc", False):
            self.network.set_hard_bc(self.sign, self.C, self.j0_aux)
            print(f"[Trainer] hard-BC2 ansatz ON  (sign={self.sign}, j0_aux={self.j0_aux:.3e})")
        
    
    def loss_pde(self, X):
        
        t = X[:,:,:1].requires_grad_()
        ksi = X[:,:,1:2].requires_grad_()
        cs_0 = X[:,:1,2:2+self.b1]
        i = X[:,:1,2+self.b1:2+self.b1+self.b2]
        
        y = self.network(t, ksi, cs_0, i)
        y_t = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y),  create_graph=True)[0]
        y_ksi = torch.autograd.grad(y, ksi, grad_outputs=torch.ones_like(y),  create_graph=True)[0]
        y_ksi_ksi = torch.autograd.grad(y_ksi, ksi, grad_outputs=torch.ones_like(y_ksi),  create_graph=True)[0]

        loss = (2*y_ksi/ksi + y_ksi_ksi) - y_t*self.Rs**2/self.Ds/self.T_tr
        return torch.mean(loss**2)
    
    def loss_ic(self, XX):
        ksi = XX[:,:,1:2]
        t = torch.zeros_like(ksi)
        cs_0 = XX[:,:1,2:2+self.b1]
        y_true = XX[:,:,-1:]
        i = XX[:,:1,2+self.b1:2+self.b1+self.b2]
        
        y = self.network(t, ksi, cs_0, i)
        loss = y_true - y

        return torch.mean(loss**2)

    def loss_bc1(self, X):
   
        t = X[:,:,:1]
        ksi = torch.zeros_like(t).requires_grad_()
        cs_0 = X[:,:1,2:2+self.b1]
        i = X[:,:1,2+self.b1:2+self.b1+self.b2]
        
        y = self.network(t, ksi, cs_0, i)
        y_ksi = torch.autograd.grad(y, ksi, grad_outputs=torch.ones_like(y),  create_graph=True)[0]
        return torch.mean(y_ksi**2)
    
    def loss_bc2(self, X):
        t = X[:,:,:1]
        ksi = torch.ones_like(t).requires_grad_()
        cs_0 = X[:,:1,2:2+self.b1]
        i = X[:,:1,2+self.b1:2+self.b1+self.b2]
        
        y = self.network(t, ksi, cs_0, i)
        y_ksi = torch.autograd.grad(y, ksi, grad_outputs=torch.ones_like(y),  create_graph=True)[0]
        loss = y_ksi + self.sign*i*self.C/self.j0_aux
        return torch.mean(loss**2)
    
    def loss_cons(self, X, n_ksi=41, n_t=4):
        """Mass conservation of the volume-averaged concentration.

        Integrating the PDE over the particle kills the diffusion term (divergence
        theorem), leaving only the surface flux; substituting the BC2 flux gives

            <y(tau)> - <y(0)> = -3*sign*i*(C/j0_aux)*(Ds*T_tr/Rs^2)*tau,
            <y> = int_0^1 3*ksi^2*y dksi

        This is an exact CONSEQUENCE of loss_pde + loss_bc2, but they enforce it
        only weakly and pointwise: the trained surrogate still violates it by
        ~0.3-18 % under load, and drifts even at i = 0 where the average must be
        exactly constant. That matters because <y> is what SOC is computed from,
        while the pointwise residuals mostly constrain the profile SHAPE - hence a
        surrogate that fits voltage (surface) well while SOC (average) drifts.

        Weight is loss_weights[3]; set it to 0 to disable this term entirely."""
        B = X.shape[0]
        cs_0 = X[:, :1, 2:2+self.b1]
        i    = X[:, :1, 2+self.b1:2+self.b1+self.b2]
        i_s  = i.reshape(B, -1)[:, :1]                      # [B,1]

        ksi = torch.linspace(1e-6, 1.0, n_ksi, device=X.device,
                             dtype=X.dtype).view(1, -1, 1).expand(B, -1, -1)
        w = 3.0 * ksi**2                                     # spherical weight
        avg0 = torch.trapezoid(w * self.network(torch.zeros_like(ksi), ksi, cs_0, i),
                               ksi, dim=1)                   # <y(0)>

        k = 3.0 * self.sign * self.C / self.j0_aux * self.Ds * self.T_tr / self.Rs**2
        t_max = X[:, :, :1].max()
        res = []
        for f in range(1, n_t + 1):
            tau = t_max * (f / n_t)
            avg = torch.trapezoid(w * self.network(torch.full_like(ksi, float(tau)),
                                                   ksi, cs_0, i), ksi, dim=1)
            res.append(avg - avg0 + k * i_s * tau)
        return torch.mean(torch.stack(res)**2)

    # ── Recursive physics-informed rollout ───────────────────────────────────

    def _linear_interp(self, cs_0: torch.Tensor, ksi_query: torch.Tensor) -> torch.Tensor:
        """
        Differentiable piecewise-linear interpolation of cs_0 at arbitrary ξ positions.

        cs_0:      (1, 1, b1) — profile values at self.sensors1 positions
        ksi_query: (1, N,  1) — query positions in [0, 1]
        Returns:   (1, N,  1)

        Gradient flows through cs_0 (indexed via torch.gather with fixed integer
        indices determined by searchsorted). The knot positions are fixed, so only
        the concentration values carry gradient.
        """
        dtype  = cs_0.dtype
        device = cs_0.device
        xi_k   = torch.as_tensor(self.sensors1, device=device, dtype=dtype)  # (b1,) sensor positions

        xi_q = ksi_query[:, :, 0].clamp(xi_k[0], xi_k[-1])  # (1, N)
        idx  = torch.searchsorted(xi_k.contiguous(), xi_q.contiguous())
        idx  = idx.clamp(1, self.b1 - 1)                     # (1, N)  lower bracket

        x0 = xi_k[idx - 1]                                   # (1, N)
        x1 = xi_k[idx]                                        # (1, N)
        alpha = (xi_q - x0) / (x1 - x0 + 1e-12)             # (1, N)

        cs_flat = cs_0[:, 0, :]                               # (B_cs, b1)
        B_q = xi_q.shape[0]
        if cs_flat.shape[0] == 1 and B_q > 1:
            cs_flat = cs_flat.expand(B_q, -1)             # (B_q, b1)
        y0 = torch.gather(cs_flat, 1, (idx - 1).clamp(0, self.b1 - 1))  # (B_q, N)
        y1 = torch.gather(cs_flat, 1,  idx      .clamp(0, self.b1 - 1)) # (B_q, N)

        return (y0 + alpha * (y1 - y0)).unsqueeze(-1)         # (1, N, 1)

    def loss_pi_rollout(
        self,
        X_pde: torch.Tensor,
        X_ic:  torch.Tensor,
        X_bc:  torch.Tensor,
        K: int  = 3,
        detach_cs: bool = False,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """
        K-step recursive physics-informed loss — no labeled data required.

        At each step k the model is called with the previous step's output as cs_0.
        The IC loss enforces cs_k(ξ, 0) = cs_{k-1}(ξ), creating a differentiable
        chain so that gradients propagate backward through the rollout via BPTT.

        X_pde, X_ic, X_bc: standard PI collocation batches (same format as
            loss_pde / loss_ic / loss_bc). Fresh batches each call give diverse
            collocation points; reusing is also fine.
        K:            number of recursive steps (K=3–5 is a good range).
        detach_cs:    if True, stop gradient through cs_0 at each advance step.
                      Use this for the first few epochs to stabilise training
                      before switching to full backprop-through-time.
                      Always True when used for validation.
        create_graph: set False for validation — skips the second-order graph
                      needed for BPTT, keeping only what is required to compute
                      the second spatial derivative (d²c/dξ² always needs one
                      intermediate graph regardless).
        """
        device = self.device
        dtype  = X_ic.dtype

        # Initial cs_0: singleton batch so it broadcasts against any trunk batch dimension.
        # model_old.py forward does out1*out2*out_trunk — out1/out2 must be singleton
        # or match trunk's batch exactly. Slicing [:1] gives (1, 1, b1).
        cs_0 = X_ic[:1, :1, 2:2 + self.b1]

        losses = []
        for k in range(K):
            # Sample a fresh current for this step from the BC batch
            # (each row of X_bc carries a different normalised current in b2 cols)
            row = k % X_bc.shape[1]
            I_k = X_bc[:1, row:row+1, 2+self.b1:2+self.b1+self.b2]  # (1, 1, b2)

            # ── PDE residual ─────────────────────────────────────────────
            t_p   = X_pde[:, :, :1].requires_grad_()
            ksi_p = X_pde[:, :, 1:2].requires_grad_()
            y_p   = self.network(t_p, ksi_p, cs_0, I_k)
            y_t   = torch.autograd.grad(y_p,  t_p,   grad_outputs=torch.ones_like(y_p),  create_graph=create_graph)[0]
            # y_k always needs create_graph=True so that y_kk (second derivative) can be computed
            y_k   = torch.autograd.grad(y_p,  ksi_p, grad_outputs=torch.ones_like(y_p),  create_graph=True)[0]
            y_kk  = torch.autograd.grad(y_k,  ksi_p, grad_outputs=torch.ones_like(y_k),  create_graph=create_graph)[0]
            pde_res = (2 * y_k / ksi_p + y_kk) - y_t * self.Rs**2 / self.Ds / self.T_tr
            l_pde = torch.mean(pde_res**2)

            # ── IC: network at t=0 must reproduce cs_0 ───────────────────
            ksi_ic  = X_ic[:, :, 1:2]
            y_ic    = self.network(torch.zeros_like(ksi_ic), ksi_ic, cs_0, I_k)
            y_ic_gt = self._linear_interp(cs_0, ksi_ic)
            l_ic = torch.mean((y_ic - y_ic_gt)**2)

            # ── BC1: dc/dξ = 0 at ξ=0 ───────────────────────────────────
            t_bc   = X_bc[:, :, :1]
            ksi_b1 = torch.zeros_like(t_bc).requires_grad_()
            y_b1   = self.network(t_bc, ksi_b1, cs_0, I_k)
            dy_b1  = torch.autograd.grad(y_b1, ksi_b1, grad_outputs=torch.ones_like(y_b1), create_graph=create_graph)[0]
            l_bc1  = torch.mean(dy_b1**2)

            # ── BC2: dc/dξ = -sign·I·C/j0 at ξ=1 ───────────────────────
            ksi_b2 = torch.ones_like(t_bc).requires_grad_()
            y_b2   = self.network(t_bc, ksi_b2, cs_0, I_k)
            dy_b2  = torch.autograd.grad(y_b2, ksi_b2, grad_outputs=torch.ones_like(y_b2), create_graph=create_graph)[0]
            l_bc2  = torch.mean((dy_b2 + self.sign * I_k * self.C / self.j0_aux)**2)

            losses.append(
                self.W_pde * l_pde +
                self.W_ic  * l_ic  +
                self.W_bc  * (l_bc1 + l_bc2)
            )

            # ── Advance: query end-of-window profile at sensor positions ──
            xi_s  = torch.as_tensor(self.sensors1, device=device, dtype=dtype).reshape(1, self.b1, 1)
            t_end = torch.ones(1, self.b1, 1, device=device, dtype=dtype)
            cs_next = self.network(t_end, xi_s, cs_0, I_k)   # (1, b1, 1)
            cs_0 = cs_next.permute(0, 2, 1)                   # (1, 1, b1)
            if detach_cs:
                cs_0 = cs_0.detach()

        return sum(losses) / K

    def loss_dd(self, XX):
        t = XX[:,:,:1]
        ksi = XX[:,:,1:2]
        cs_0 = XX[:,:,2:2+self.b1]
        i = XX[:,:,2+self.b1:2+self.b1+self.b2]
        y_true = XX[:,:,-1:]

        y = self.network(t, ksi, cs_0, i)

        loss = y_true - y
        
        return torch.mean(loss**2)
        
    def replace_optimizer(self, lr_lbfgs = 1e-2):
        self.optimizer = torch.optim.LBFGS(self.network.parameters(),
                                    lr = lr_lbfgs,
                                    max_iter=100,
                                    tolerance_grad=1e-6,
                                    tolerance_change=1e-9,
                                    line_search_fn = "strong_wolfe")
        self.tracker.log_every = 2
        self.change_opt = False
        
    def evaluation(self, X_eval):
        
        loss_eval = self.loss_dd(X_eval).item()
        
        print('Evaluation loss on an unseen set of data: {:10.2e} (MSE)'.format(loss_eval))
        
        return loss_eval
        
    def predict(self, res, dT, Cs_0, Curr):
        
        T = np.ones([1,res,1])*dT
        Ksi = np.zeros((1,res,1))
        Ksi[0,:,0] = np.linspace(0,1,res)
        
        t = torch.tensor(T,dtype=torch.float32)
        ksi = torch.tensor(Ksi,dtype=torch.float32)
        cs_0 =torch.tensor(Cs_0,dtype=torch.float32)
        curr =torch.tensor(np.array([[Curr]]),dtype=torch.float32)

        y = self.network(t, ksi, cs_0, curr)
        return y.detach().numpy()[0,:,:1]
    