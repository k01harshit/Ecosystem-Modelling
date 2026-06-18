"""
models/neural_ode.py

Neural ODE module for ecological stability prediction.

Key idea:
  The GNN encoder produces node embeddings h_i.
  A pairwise MLP maps (h_i, h_j, edge_attr) → interaction coefficient â_ij.
  These coefficients populate a Lotka-Volterra interaction matrix Â.
  A Neural ODE integrates the LV system forward in time:
      dx_i/dt = x_i * (r_i + Σ_j â_ij * x_j)
  Stability is inferred from the long-run trajectory.

This gives mechanistic interpretability: â_ij can be compared
to measured ecological interaction strengths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

try:
    from torchdiffeq import odeint
    ODE_AVAILABLE = True
except ImportError:
    ODE_AVAILABLE = False

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import EMBED_DIM, ODE_TIME_STEPS, ODE_T_END, ODE_SOLVER


class InteractionMatrixPredictor(nn.Module):
    """
    Predicts pairwise LV interaction coefficient â_ij from node embeddings.

    For each (i,j) pair, concatenate h_i and h_j and pass through MLP.
    Output is a scalar interaction coefficient (can be positive or negative).
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, embed_dim),   # +1 for edge weight
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Tanh(),   # bound output to (-1, 1) for stability
        )

    def forward(self, h_i: Tensor, h_j: Tensor,
                edge_weight: Tensor) -> Tensor:
        """
        h_i, h_j     : (K, EMBED_DIM) — embeddings of K interacting pairs
        edge_weight  : (K, 1)         — interaction weight from data
        Returns a_ij : (K,)           — predicted interaction coefficients
        """
        pair = torch.cat([h_i, h_j, edge_weight], dim=-1)
        return self.mlp(pair).squeeze(-1)


class LotkaVolterraODE(nn.Module):
    """
    ODE function for the generalized Lotka-Volterra system:
        dx_i/dt = x_i * (r_i + Σ_j A_ij * x_j)

    This is passed to torchdiffeq.odeint as the dynamics function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, t: Tensor, x: Tensor,
                A: Tensor, r: Tensor) -> Tensor:
        """
        t : scalar time (unused in autonomous system, required by odeint)
        x : (N,) population vector
        A : (N, N) interaction matrix
        r : (N,) intrinsic growth rates
        Returns dx/dt : (N,)
        """
        dx = x * (r + A @ x)
        # Clip to prevent population explosion
        dx = torch.clamp(dx, -50.0, 50.0)
        return dx


class EcologicalDynamicsModule(nn.Module):
    """
    Full dynamics module:
      1. Predict interaction matrix Â from node embeddings
      2. Simulate LV system with Neural ODE
      3. Return trajectory and stability indicators
    """

    def __init__(self, embed_dim: int = EMBED_DIM,
                 t_end: float = ODE_T_END,
                 n_steps: int = ODE_TIME_STEPS,
                 solver: str = ODE_SOLVER):
        super().__init__()
        self.interaction_predictor = InteractionMatrixPredictor(embed_dim)
        self.lv_ode = LotkaVolterraODE()
        self.t_span = torch.linspace(0, t_end, n_steps)
        self.solver = solver

    def build_interaction_matrix(self, node_emb: Tensor,
                                  edge_index: Tensor,
                                  edge_attr: Tensor,
                                  n_nodes: int) -> Tensor:
        """
        Build the N×N interaction matrix from edge-wise predictions.
        Diagonal set to -1.0 (self-regulation, standard in LV).
        Off-diagonal set to predicted â_ij for observed edges, 0 otherwise.
        """
        A = torch.zeros(n_nodes, n_nodes, device=node_emb.device)

        if edge_index.size(1) > 0:
            src, tgt = edge_index[0], edge_index[1]
            h_src = node_emb[src]
            h_tgt = node_emb[tgt]
            w     = edge_attr[:, 0:1]            # (E, 1)
            a_ij  = self.interaction_predictor(h_src, h_tgt, w)   # (E,)
            A[src, tgt] = a_ij

        # Self-regulation: species-specific, not fixed -1.
        # Standard LV theory: d_ii = -a_ii where a_ii reflects intraspecific
        # competition. High trophic level species have weaker self-regulation
        # (less intraspecific competition), making them more vulnerable.
        # We derive self-regulation from node embeddings via a learnable scalar,
        # bounded to (-2, -0.05) to keep dynamics bounded but allow instability.
        diag_vals = torch.sigmoid(
            node_emb.norm(dim=-1, keepdim=False)
        ) * (-1.8) - 0.05    # range (-1.85, -0.05)
        A.diagonal().copy_(diag_vals.detach())
        return A

    def simulate(self, A: Tensor, r: Tensor,
                 x0: Tensor = None) -> Tensor:
        """
        Simulate LV dynamics.

        A  : (N, N) interaction matrix
        r  : (N,) intrinsic growth rates
        x0 : (N,) initial populations (default: all 0.5)

        Returns trajectory: (T, N) population values over time.
        """
        N = A.size(0)
        if x0 is None:
            x0 = torch.full((N,), 0.5, device=A.device)
        x0 = x0.clamp(min=0.01)

        t_span = self.t_span.to(A.device)

        if ODE_AVAILABLE:
            # Use torchdiffeq for differentiable integration
            def ode_func(t, x):
                x_clamped = x.clamp(min=0.0)
                return self.lv_ode(t, x_clamped, A, r)

            try:
                traj = odeint(ode_func, x0, t_span,
                              method=self.solver,
                              rtol=1e-3, atol=1e-4)  # (T, N)
            except Exception:
                traj = self._euler_integrate(x0, A, r, t_span)
        else:
            traj = self._euler_integrate(x0, A, r, t_span)

        return traj.clamp(min=0.0)

    def forward(self, node_emb: Tensor, edge_index: Tensor,
                edge_attr: Tensor, growth: Tensor,
                x0: Tensor = None) -> dict:
        """
        Full forward pass: encode → build A → simulate → analyse.

        Returns dict:
            A          : (N, N) predicted interaction matrix
            trajectory : (T, N) population over time
            final_pop  : (N,) final populations
            extinct    : (N,) boolean — population < 0.01 at end
            stable     : bool — did system reach equilibrium?
            resilience : float — stability score [0, 1]
        """
        N = node_emb.size(0)
        A = self.build_interaction_matrix(node_emb, edge_index, edge_attr, N)
        traj = self.simulate(A, growth, x0)

        final_pop = traj[-1]
        extinct   = (final_pop < 0.01)

        # Stability: compare last 20% of trajectory variance
        tail = traj[int(0.8 * len(traj)):]
        variance    = tail.var(dim=0).mean().item()
        resilience  = float(np.exp(-variance))

        # Spectral stability: max real part of Jacobian eigenvalues
        # With species-specific self-regulation, this CAN be positive
        x_eq    = final_pop.detach()
        J       = torch.diag(x_eq) @ A.detach()
        eigvals = torch.linalg.eigvals(J)
        max_re  = eigvals.real.max().item()
        stable  = max_re < 0

        # Peak deviation: max population excursion from initial conditions
        # Captures transient instability even if system eventually settles.
        # High peak_dev = destabilising invasion even if it recovers.
        x0_ref     = traj[0].detach()
        peak_dev   = (traj.detach() - x0_ref.unsqueeze(0)).abs().max().item()

        # Invader dominance: final population of invader (last node)
        # relative to mean native population
        invader_final = final_pop[-1].item()
        native_mean   = final_pop[:-1].mean().item() if final_pop.size(0) > 1 else 1.0
        invader_ratio = float(np.clip(invader_final / (native_mean + 1e-6), 0, 20))

        return {
            "A":             A,
            "trajectory":    traj,
            "final_pop":     final_pop,
            "extinct":       extinct,
            "stable":        stable,
            "resilience":    resilience,
            "max_eigenvalue": max_re,
            "peak_dev":      peak_dev,      # NEW: transient instability
            "invader_ratio": invader_ratio,  # NEW: invader dominance
        }

    @staticmethod
    def _euler_integrate(x0: Tensor, A: Tensor, r: Tensor,
                         t_span: Tensor) -> Tensor:
        """Simple Euler integration fallback."""
        dt    = (t_span[-1] - t_span[0]) / (len(t_span) - 1)
        traj  = [x0]
        x     = x0.clone()
        for _ in range(len(t_span) - 1):
            dx = x * (r + A @ x)
            dx = torch.clamp(dx, -50.0, 50.0)
            x  = (x + dt * dx).clamp(min=0.0)
            traj.append(x)
        return torch.stack(traj)
