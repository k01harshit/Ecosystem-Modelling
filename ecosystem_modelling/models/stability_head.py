"""
models/stability_head.py

Wraps encoder + ODE dynamics into a single forward pass that returns
the graph embeddings and ODE outputs needed by DisruptionScorer.

StabilityHead and SpeciesRiskHead have been removed — they were part of
the old supervised fine-tuning pipeline which is no longer used.
"""

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import EMBED_DIM


class FullStabilityModel(nn.Module):
    """
    Wraps encoder + dynamics into one forward pass.
    Returns graph embeddings and ODE outputs for the DisruptionScorer.
    No supervised heads — scoring is done externally via fixed weights.
    """

    def __init__(self, encoder, dynamics_module, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.encoder  = encoder
        self.dynamics = dynamics_module

    def forward(self, pre_graph, post_graph) -> dict:
        pre_node_emb,  pre_graph_emb  = self.encoder(
            pre_graph.x, pre_graph.edge_index, pre_graph.edge_type)
        post_node_emb, post_graph_emb = self.encoder(
            post_graph.x, post_graph.edge_index, post_graph.edge_type)

        ode_out = self.dynamics(
            post_node_emb,
            post_graph.edge_index,
            post_graph.edge_attr,
            post_graph.growth,
        )

        soft_extinct       = torch.sigmoid(-400.0 * (ode_out["final_pop"] - 0.01))
        ode_out["extinct"] = soft_extinct

        return {
            "pre_graph_emb":  pre_graph_emb,
            "post_graph_emb": post_graph_emb,
            "ode":            ode_out,
        }
