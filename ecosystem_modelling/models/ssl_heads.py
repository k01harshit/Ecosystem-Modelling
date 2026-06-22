"""
models/ssl_heads.py

Three self-supervised learning objectives used for pretraining the encoder.

1. MaskedEdgeReconstruction
   - Reconstruct masked edges from node embeddings
   - Loss: BCE (existence) + CE (type) + MSE (weight)

2. TrophicLevelPrediction
   - Predict each node's trophic level from its embedding
   - Loss: MSE
   - This is self-supervised because trophic level is computable
     from the graph structure itself (no external labels needed)

3. GraphContrastiveLoss
   - Two augmented views of the same graph should have similar embeddings
   - Uses InfoNCE (NT-Xent) loss
   - Positive pair: same graph, different augmentation
   - Negatives: all other graphs in the batch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import EMBED_DIM, SSL_CONTRASTIVE_TAU, SSL_TROPHIC_WEIGHT


class MaskedEdgeReconstruction(nn.Module):
    """
    Reconstruct masked (removed) edges from node embeddings.

    Given:
        node_emb      : (N, EMBED_DIM) — embeddings after encoding masked graph
        masked_edges  : (2, K) — source/target indices of masked edges
        masked_labels : (K, 2) — [weight, edge_type] for each masked edge

    Returns scalar loss.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, num_types: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
        )
        self.exist_head  = nn.Linear(embed_dim // 2, 1)
        self.type_head   = nn.Linear(embed_dim // 2, num_types)
        self.weight_head = nn.Linear(embed_dim // 2, 1)

    def forward(self, node_emb: Tensor,
                masked_edges:  Tensor,
                masked_labels: Tensor,
                all_edge_index: Tensor = None) -> Tensor:
        """
        node_emb       : (N, D)
        masked_edges   : (2, K)
        masked_labels  : (K, 2)  col0=weight, col1=edge_type (float)
        all_edge_index : (2, E_keep)  — kept edges (for negative sampling)
        """
        if masked_edges.size(1) == 0:
            return torch.tensor(0.0, requires_grad=True,
                                device=node_emb.device)

        src_emb = node_emb[masked_edges[0]]   # (K, D)
        tgt_emb = node_emb[masked_edges[1]]   # (K, D)
        pair    = torch.cat([src_emb, tgt_emb], dim=-1)
        z       = self.mlp(pair)

        # --- Edge type loss (classification) ----------------------------------
        type_targets = masked_labels[:, 1].long()
        type_loss    = F.cross_entropy(self.type_head(z), type_targets)

        # --- Weight regression loss -------------------------------------------
        w_pred   = F.softplus(self.weight_head(z)).squeeze(-1)
        w_target = masked_labels[:, 0]
        weight_loss = F.mse_loss(w_pred, w_target)

        # --- Existence loss: positive edges + random negatives ----------------
        K  = masked_edges.size(1)
        N  = node_emb.size(0)
        # sample K random negative pairs
        neg_src = torch.randint(0, N, (K,), device=node_emb.device)
        neg_tgt = torch.randint(0, N, (K,), device=node_emb.device)
        neg_pair  = torch.cat([node_emb[neg_src], node_emb[neg_tgt]], dim=-1)
        neg_z     = self.mlp(neg_pair)

        pos_logit = self.exist_head(z).squeeze(-1)
        neg_logit = self.exist_head(neg_z).squeeze(-1)
        exist_logits = torch.cat([pos_logit, neg_logit])
        exist_labels = torch.cat([
            torch.ones(K,  device=node_emb.device),
            torch.zeros(K, device=node_emb.device),
        ])
        exist_loss = F.binary_cross_entropy_with_logits(exist_logits,
                                                         exist_labels)

        loss = exist_loss + type_loss + 0.5 * weight_loss
        return loss


class TrophicLevelPrediction(nn.Module):
    """
    Predict trophic level of each node from its embedding.
    Self-supervised: trophic level is derived from graph structure,
    not measured externally.
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, node_emb: Tensor, trophic_targets: Tensor) -> Tensor:
        """
        node_emb        : (N, D)
        trophic_targets : (N,)   — ground truth trophic levels
        """
        pred = self.head(node_emb).squeeze(-1)   # (N,)
        # Normalise targets to [0,1] range for stable training
        t_norm = trophic_targets / 6.0
        return F.mse_loss(pred, t_norm)


class GraphContrastiveLoss(nn.Module):
    """
    NT-Xent contrastive loss over graph-level embeddings.

    Given a batch of (view_a, view_b) pairs:
      - view_a[i] and view_b[i] are positive pairs (same ecosystem)
      - All cross-pairs are negatives

    Temperature parameter τ controls the sharpness of the distribution.
    """

    def __init__(self, tau: float = SSL_CONTRASTIVE_TAU,
                 embed_dim: int = EMBED_DIM):
        super().__init__()
        self.tau = tau
        # Small projection head (SimCLR style — improves representation quality)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
        )

    def forward(self, z_a: Tensor, z_b: Tensor) -> Tensor:
        """
        z_a : (B, D) graph embeddings from view A
        z_b : (B, D) graph embeddings from view B
        """
        B = z_a.size(0)
        if B < 2:
            return torch.tensor(0.0, requires_grad=True, device=z_a.device)

        p_a = F.normalize(self.projector(z_a), dim=-1)  # (B, D/2)
        p_b = F.normalize(self.projector(z_b), dim=-1)  # (B, D/2)

        # Similarity matrix: (2B, 2B)
        reps    = torch.cat([p_a, p_b], dim=0)          # (2B, D/2)
        sim_mat = torch.mm(reps, reps.t()) / self.tau   # (2B, 2B)

        # Mask out self-similarity
        mask = torch.eye(2 * B, dtype=torch.bool, device=z_a.device)
        sim_mat = sim_mat.masked_fill(mask, float("-inf"))

        # Positive pair indices: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z_a.device),
            torch.arange(0, B,     device=z_a.device),
        ])  # (2B,)

        loss = F.cross_entropy(sim_mat, labels)
        return loss


class SSLHead(nn.Module):
    """
    Combines all three SSL objectives into one module.
    Weights are tunable but default to equal contribution.
    """

    def __init__(self,
                 embed_dim: int = EMBED_DIM,
                 w_masked: float = 1.0,
                 w_trophic: float = SSL_TROPHIC_WEIGHT,
                 w_contrastive: float = 1.0):
        super().__init__()
        self.masked_edge  = MaskedEdgeReconstruction(embed_dim)
        self.trophic_pred = TrophicLevelPrediction(embed_dim)
        self.contrastive  = GraphContrastiveLoss(embed_dim=embed_dim)
        self.w_m = w_masked
        self.w_t = w_trophic
        self.w_c = w_contrastive

    def forward(self,
                node_emb_masked: Tensor,
                masked_edges: Tensor,
                masked_labels: Tensor,
                trophic_targets: Tensor,
                graph_emb_a: Tensor,
                graph_emb_b: Tensor) -> dict:
        """
        Returns dict with individual losses and total loss.
        """
        l_masked = self.masked_edge(node_emb_masked, masked_edges,
                                    masked_labels)
        l_trophic = self.trophic_pred(node_emb_masked, trophic_targets)
        l_contra  = self.contrastive(graph_emb_a, graph_emb_b)

        total = (self.w_m * l_masked +
                 self.w_t * l_trophic +
                 self.w_c * l_contra)

        return {
            "total":       total,
            "masked_edge": l_masked.item(),
            "trophic":     l_trophic.item(),
            "contrastive": l_contra.item(),
        }
