"""Muon: momentum orthogonalized by quintic Newton-Schulz iteration for hidden 2-D matrices.

Update scale follows Moonshot AI ("Muon is Scalable for LLM Training"): 0.2 * sqrt(max(rows, cols)),
matching AdamW's RMS update magnitude so both optimizers share one learning rate and schedule.
"""

from __future__ import annotations

import os
from typing import Iterable

import torch

_HIDDEN_SUBSTRINGS = (
    "w_state",
    "mid",
    "out",
    "w_recall",
    "write",
    "self_attn.in_proj_weight",
    "self_attn.out_proj",
    "linear1",
    "linear2",
    "wqkv",
    ".wo.",
    ".wv.",
    "w_up",
    "w_down",
)


@torch.no_grad()
def orthogonalize_batch(G: torch.Tensor, steps: int = 5, bf16: bool = True) -> torch.Tensor:
    """Approximate U V^T of G = U S V^T over a batch [n, rows, cols] of same-shape matrices."""
    a, b, c = 3.4445, -4.7750, 2.0315
    use_bf16 = bf16 and G.is_cuda
    X = G.bfloat16() if use_bf16 else G.float()
    tr = X.shape[1] > X.shape[2]
    if tr:
        X = X.transpose(1, 2)
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        X = a * X + (b * A + c * A @ A) @ X
    X = X.transpose(1, 2) if tr else X
    return X.float()


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        bf16: bool | None = None,
    ):
        if bf16 is None:
            bf16 = bool(os.environ.get("MUON_BF16", "1") != "0")
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=weight_decay,
                ns_steps=ns_steps,
                bf16=bf16,
            ),
        )
        self._ortho_fn = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            ps = [p for p in g["params"] if p.grad is not None]
            if not ps:
                continue
            on_cuda = ps[0].is_cuda
            if on_cuda and self._ortho_fn is None:
                eager = bool(os.environ.get("MUON_EAGER"))
                self._ortho_fn = (
                    orthogonalize_batch
                    if eager
                    else torch.compile(orthogonalize_batch, mode="reduce-overhead")
                )
            for p in ps:
                if "buf" not in self.state[p]:
                    self.state[p]["buf"] = torch.zeros_like(p)
            grads = [p.grad for p in ps]
            bufs = [self.state[p]["buf"] for p in ps]
            torch._foreach_mul_(bufs, g["momentum"])
            torch._foreach_add_(bufs, grads)
            us = torch._foreach_add(grads, bufs, alpha=g["momentum"]) if g["nesterov"] else bufs

            jobs, groups = [], {}
            for p, u in zip(ps, us):
                k = getattr(p, "_muon_split", 1)
                parts = u.split(k, 0) if isinstance(k, list) else u.chunk(k, 0)
                slots = []
                for chunk in parts:
                    sh = tuple(chunk.shape)
                    lst = groups.setdefault(sh, [])
                    slots.append((sh, len(lst)))
                    lst.append(chunk)
                jobs.append(slots)

            use_bf16 = g["bf16"] and on_cuda
            ortho = self._ortho_fn if on_cuda else orthogonalize_batch
            done = {
                sh: ortho(torch.stack(lst), g["ns_steps"], use_bf16).clone() * (0.2 * max(sh) ** 0.5)
                for sh, lst in groups.items()
            }
            upd = [
                torch.cat([done[sh][i] for sh, i in slots]) if len(slots) > 1 else done[slots[0][0]][slots[0][1]]
                for slots in jobs
            ]
            if g["weight_decay"]:
                torch._foreach_mul_(ps, 1.0 - g["lr"] * g["weight_decay"])
            torch._foreach_add_(ps, upd, alpha=-g["lr"])
        return loss


def split_params(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split parameters into (hidden 2-D matrices for Muon, remaining parameters for AdamW)."""
    mu, ad = [], []
    for name, p in model.named_parameters():
        if p.ndim == 2 and any(h in name for h in _HIDDEN_SUBSTRINGS) and not name.startswith("ngrams"):
            if "in_proj_weight" in name or "wqkv" in name:
                p._muon_split = 3
            mu.append(p)
        else:
            ad.append(p)
    return mu, ad
