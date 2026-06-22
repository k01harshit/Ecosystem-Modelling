"""
training/pretrain.py  — v2

Key improvements over v1:
  1. Linear LR warmup before cosine decay (prevents early instability)
  2. Gradient accumulation (effective larger batch for contrastive loss)
  3. Per-graph gradient steps instead of epoch-level accumulation
     — fixes the masked edge spike pattern seen in v1
  4. Gradient clipping per step
  5. Better logging (loss EMA instead of raw average)
"""

import torch
import torch.optim as optim
import logging
from typing import List
from torch_geometric.data import Data

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import (SSL_PRETRAIN_EPOCHS, SSL_LR, SSL_MASK_RATE, SEED,
                     SSL_WARMUP_EPOCHS)
from data.augmentation import augment_graph, mask_edges_for_reconstruction

logger = logging.getLogger(__name__)


def pretrain(encoder, ssl_head, graphs: List[Data],
             epochs: int = SSL_PRETRAIN_EPOCHS,
             lr:     float = SSL_LR,
             device: str   = "cpu",
             seed:   int   = None) -> dict:
    """
    SSL pretraining with warmup + cosine decay.

    Structural change from v1:
      In v1, contrastive loss was computed once at the end of each epoch
      over ALL graphs accumulated. This caused gradient staleness.

      In v2, we process graphs in mini-batches of size CONTRASTIVE_BATCH.
      Each mini-batch does a full forward+backward including contrastive loss.
      This gives more frequent, fresher gradient updates.
    """
    torch.manual_seed(seed if seed is not None else SEED)
    encoder.to(device)
    ssl_head.to(device)

    params = list(encoder.parameters()) + list(ssl_head.parameters())
    opt    = optim.AdamW(params, lr=lr, weight_decay=1e-5)

    # Warmup + cosine schedule
    def lr_lambda(epoch):
        if epoch < SSL_WARMUP_EPOCHS:
            return float(epoch + 1) / float(SSL_WARMUP_EPOCHS)
        progress = (epoch - SSL_WARMUP_EPOCHS) / max(1, epochs - SSL_WARMUP_EPOCHS)
        return 0.1 + 0.9 * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    history = {"total": [], "masked_edge": [], "trophic": [], "contrastive": []}

    CONTRASTIVE_BATCH = 16   # process this many graphs per contrastive step
    EMA_ALPHA         = 0.1  # smoothing for logged losses

    ema = {"total": None, "masked_edge": None, "trophic": None, "contrastive": None}

    for epoch in range(1, epochs + 1):
        encoder.train()
        ssl_head.train()

        epoch_losses = {"total": 0.0, "masked_edge": 0.0,
                        "trophic": 0.0, "contrastive": 0.0}
        n_steps = 0

        # Shuffle graphs each epoch
        perm   = torch.randperm(len(graphs)).tolist()
        shuffled = [graphs[i] for i in perm]

        # Process in contrastive mini-batches
        for batch_start in range(0, len(shuffled), CONTRASTIVE_BATCH):
            batch = shuffled[batch_start: batch_start + CONTRASTIVE_BATCH]
            if len(batch) < 2:
                continue

            opt.zero_grad()
            batch_z_a, batch_z_b = [], []
            batch_masked  = 0.0
            batch_trophic = 0.0

            for g in batch:
                g = g.to(device)

                # Augmented views
                view_a, view_b = augment_graph(g)
                view_a = view_a.to(device)
                view_b = view_b.to(device)

                # Masked graph
                masked_g, masked_edges, masked_labels = \
                    mask_edges_for_reconstruction(g, mask_rate=SSL_MASK_RATE)
                masked_g      = masked_g.to(device)
                masked_edges  = masked_edges.to(device)
                masked_labels = masked_labels.to(device)

                # Encode
                node_emb_masked, _ = encoder(
                    masked_g.x, masked_g.edge_index, masked_g.edge_type)
                _, z_a = encoder(view_a.x, view_a.edge_index, view_a.edge_type)
                _, z_b = encoder(view_b.x, view_b.edge_index, view_b.edge_type)

                batch_z_a.append(z_a)
                batch_z_b.append(z_b)

                # Per-graph losses (accumulate, don't backward yet)
                l_m = ssl_head.masked_edge(node_emb_masked, masked_edges,
                                            masked_labels)
                l_t = ssl_head.trophic_pred(node_emb_masked,
                                             g.trophic.to(device))
                per_graph_loss = (ssl_head.w_m * l_m + ssl_head.w_t * l_t) \
                                  / len(batch)
                per_graph_loss.backward(retain_graph=False)

                batch_masked  += l_m.item()
                batch_trophic += l_t.item()

            # Contrastive loss over the mini-batch
            z_a_cat = torch.cat(batch_z_a, dim=0)
            z_b_cat = torch.cat(batch_z_b, dim=0)
            l_c     = ssl_head.contrastive(z_a_cat, z_b_cat)
            (ssl_head.w_c * l_c).backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()

            B = len(batch)
            step_losses = {
                "masked_edge": batch_masked / B,
                "trophic":     batch_trophic / B,
                "contrastive": l_c.item(),
                "total":       batch_masked / B + batch_trophic / B + l_c.item(),
            }

            for k in epoch_losses:
                epoch_losses[k] += step_losses[k]
            n_steps += 1

        sched.step()

        if n_steps > 0:
            for k in epoch_losses:
                epoch_losses[k] /= n_steps

        # EMA smoothing for logging
        for k in ema:
            if ema[k] is None:
                ema[k] = epoch_losses[k]
            else:
                ema[k] = EMA_ALPHA * epoch_losses[k] + (1 - EMA_ALPHA) * ema[k]

        for k in history:
            history[k].append(epoch_losses[k])

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"total={ema['total']:.4f} | "
                f"masked={ema['masked_edge']:.4f} | "
                f"trophic={ema['trophic']:.4f} | "
                f"contrastive={ema['contrastive']:.4f} | "
                f"lr={sched.get_last_lr()[0]*lr:.6f}"
            )

    return history
