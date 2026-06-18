"""
models/encoder.py

Heterogeneous Graph Attention Network (HeteroGAT) encoder.

Architecture:
  - Separate linear projections per edge type (R-GCN style)
  - Multi-head attention for neighbor aggregation (GAT style)
  - Skip connections + LayerNorm
  - Final MLP readout to EMBED_DIM

Input:  PyG Data with x (N, NODE_FEATURE_DIM), edge_index, edge_attr, edge_type
Output: node embeddings (N, EMBED_DIM)
        graph embedding (1, EMBED_DIM) via mean pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv, global_mean_pool

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import (NODE_FEATURE_DIM, HIDDEN_DIM, EMBED_DIM,
                     NUM_HEADS, NUM_GNN_LAYERS, NUM_EDGE_TYPES)


class HeteroGATLayer(nn.Module):
    """
    One message-passing layer with per-edge-type weights (R-GCN)
    and attention (GAT).

    For each edge type t:
      - Project source node features: h_s^(t) = W_t * h_s
      - Run GAT attention with projected features
      - Sum contributions from all edge types
    """

    def __init__(self, in_dim: int, out_dim: int,
                 num_heads: int, num_edge_types: int):
        super().__init__()
        self.num_edge_types = num_edge_types
        self.out_dim = out_dim
        self.num_heads = num_heads
        head_dim = out_dim // num_heads

        # Per-edge-type source projections
        self.edge_proj = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for _ in range(num_edge_types)
        ])

        # Shared GAT conv (attention computed on projected features)
        self.gat = GATConv(
            in_channels  = out_dim,
            out_channels = head_dim,
            heads        = num_heads,
            concat       = True,
            add_self_loops = False,
        )

        self.norm = nn.LayerNorm(out_dim)
        self.skip = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_type: Tensor) -> Tensor:
        """
        x          : (N, in_dim)
        edge_index : (2, E)
        edge_type  : (E,)  integer in [0, num_edge_types)
        """
        # Aggregate per-edge-type projected messages
        aggregated = torch.zeros(x.size(0), self.out_dim, device=x.device)

        for t in range(self.num_edge_types):
            mask = (edge_type == t)
            if mask.sum() == 0:
                continue
            ei_t = edge_index[:, mask]
            x_proj = self.edge_proj[t](x)       # project ALL nodes for type t
            msg = self.gat(x_proj, ei_t)         # (N, out_dim)
            aggregated = aggregated + msg

        # Skip connection + norm
        out = self.norm(aggregated + self.skip(x))
        return F.gelu(out)


class EcologicalEncoder(nn.Module):
    """
    Full GNN encoder: stacks HeteroGATLayer × NUM_GNN_LAYERS.

    forward() returns:
        node_emb  (N, EMBED_DIM)
        graph_emb (B, EMBED_DIM)  — mean-pooled over nodes per graph
    """

    def __init__(self,
                 in_dim:          int = NODE_FEATURE_DIM,
                 hidden_dim:      int = HIDDEN_DIM,
                 embed_dim:       int = EMBED_DIM,
                 num_heads:       int = NUM_HEADS,
                 num_layers:      int = NUM_GNN_LAYERS,
                 num_edge_types:  int = NUM_EDGE_TYPES,
                 dropout:         float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # GNN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            d_in  = hidden_dim
            d_out = hidden_dim if i < num_layers - 1 else embed_dim
            self.layers.append(
                HeteroGATLayer(d_in, d_out, num_heads, num_edge_types)
            )

        self.dropout = nn.Dropout(dropout)

        # Final MLP after pooling
        self.graph_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_type: Tensor,
                batch: Tensor = None) -> tuple:
        """
        x          : (N, in_dim)
        edge_index : (2, E)
        edge_type  : (E,)
        batch      : (N,) graph membership (for batched graphs)
        """
        h = F.gelu(self.input_proj(x))
        h = self.dropout(h)

        for layer in self.layers:
            h = layer(h, edge_index, edge_type)
            h = self.dropout(h)

        # node_emb: (N, EMBED_DIM)
        node_emb = h

        # graph_emb: mean pool over nodes
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        graph_emb_raw = global_mean_pool(h, batch)       # (B, EMBED_DIM)
        graph_emb     = self.graph_mlp(graph_emb_raw)    # (B, EMBED_DIM)

        return node_emb, graph_emb


class EdgePredictor(nn.Module):
    """
    Predicts edge existence and type between a pair of node embeddings.

    Used for:
      1. Masked edge reconstruction (SSL objective)
      2. Predicting invasive species connections
    """

    def __init__(self, embed_dim: int = EMBED_DIM,
                 num_edge_types: int = NUM_EDGE_TYPES):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
        )
        self.exist_head = nn.Linear(embed_dim // 2, 1)      # binary: edge exists?
        self.type_head  = nn.Linear(embed_dim // 2, num_edge_types)
        self.weight_head = nn.Linear(embed_dim // 2, 1)     # continuous weight

    def forward(self, h_src: Tensor, h_tgt: Tensor) -> dict:
        """
        h_src, h_tgt : (K, EMBED_DIM) — embeddings of K source/target pairs
        Returns dict with keys: exist_logit, type_logit, weight
        """
        pair = torch.cat([h_src, h_tgt], dim=-1)   # (K, 2*EMBED_DIM)
        z    = self.mlp(pair)
        return {
            "exist_logit": self.exist_head(z).squeeze(-1),   # (K,)
            "type_logit":  self.type_head(z),                # (K, num_types)
            "weight":      F.softplus(self.weight_head(z)).squeeze(-1),  # (K,)
        }
