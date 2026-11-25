"""Train a Concordance byte-level hybrid reader on enwik8 with Muon + AdamW.

Presets reproduce the published 13.7M-parameter runs (~71-85 min on a single RTX 3060):
    g4k : 2 global attention layers (layers 3 & 6) -> 1.084 static / 1.023 live bpb on enwik8 test
    w4k : pure non-quadratic (O(1) per byte in context length) -> 1.136 static / 1.052 live bpb
    s4k : chunk-selected sparse far attention in layers 3 & 6

Example:
    python scripts/train.py --preset g4k
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concordance.model import Hybrid, preset_config
from concordance.muon import Muon, split_params


class CombinedOptimizer:
    """Muon for hidden 2-D matrices + AdamW for embeddings, norms, corpus projections and head."""

    def __init__(self, mu: list[torch.nn.Parameter], ad: list[torch.nn.Parameter], lr: float, wd: float):
        ad_groups = [
            {"params": [p for p in ad if p.ndim >= 2], "weight_decay": wd},
            {"params": [p for p in ad if p.ndim < 2], "weight_decay": 0.0},
        ]
        self.opts = [
            Muon(mu, lr=lr, weight_decay=wd, bf16=True),
            torch.optim.AdamW(ad_groups, lr=lr, betas=(0.9, 0.95)),
        ]
        self.param_groups = [g for opt in self.opts for g in opt.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.opts:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.opts:
            opt.step()


def load_memory_to_gpu(path: str, device: str) -> torch.Tensor:
    mm = np.load(path, mmap_mode="r")
    out = torch.empty(mm.shape, dtype=torch.uint8, device=device)
    for i in range(mm.shape[0]):
        out[i] = torch.from_numpy(np.ascontiguousarray(mm[i])).to(device)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a Concordance byte-level reader.")
    ap.add_argument("--preset", choices=["g4k", "w4k", "s4k"], default="g4k")
    ap.add_argument("--data", default="data/enwik8")
    ap.add_argument("--mem", default="data/corpus_mem2.npy")
    ap.add_argument("--mem-frozen", default="data/corpus_mem2_val_frozen.npy")
    ap.add_argument("--no-corpus", action="store_true", help="Train reader without corpus memory (ablation)")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--eval-windows", type=int, default=96)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--out", default="", help="Output checkpoint path (default: checkpoints/concordance_<preset>.pt)")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-flex", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Training script expects a CUDA GPU.")

    dev = "cuda"
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = preset_config(args.preset)
    cfg["ctx"] = args.ctx
    if args.no_corpus:
        cfg["corpus_orders"] = 0

    model = Hybrid(**cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"preset={args.preset}  params={n_params / 1e6:.2f}M  MACs/byte={model.macs_per_byte(args.ctx) / 1e6:.2f}M")
    print(f"cfg={model.config()}", flush=True)

    if not args.no_flex:
        model.flex = True
        model._block_mask(args.ctx, torch.device(dev))

    compiled = model if args.no_compile else torch.compile(model)

    def forward_logits(x: torch.Tensor, cf: torch.Tensor | None) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = compiled(x, cf) if cf is not None else compiled(x)
        return out.float()

    mu, ad = split_params(model)
    print(
        f"muon={sum(p.numel() for p in mu) / 1e6:.2f}M ({len(mu)} matrices)  "
        f"adamw={sum(p.numel() for p in ad) / 1e6:.2f}M",
        flush=True,
    )
    opt = CombinedOptimizer(mu, ad, lr=args.lr, wd=args.wd)

    raw = np.fromfile(args.data, dtype=np.uint8, count=95_000_000)
    train_data = torch.from_numpy(raw[:90_000_000].copy()).to(dev)
    val_data = torch.from_numpy(raw[90_000_000:95_000_000].copy()).to(dev)

    cmem = cmem_frozen = None
    if cfg["corpus_orders"]:
        cmem = load_memory_to_gpu(args.mem, dev)
        cmem_frozen = load_memory_to_gpu(args.mem_frozen, dev)

    def make_batch(src: torch.Tensor, offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        idx = offs[:, None] + torch.arange(args.ctx + 1, device=dev)
        w = src[idx].long()
        return w[:, :-1], w[:, 1:]

    def make_cf(arr: torch.Tensor | None, base: int, offs: torch.Tensor) -> torch.Tensor | None:
        if arr is None:
            return None
        idx = base + offs[:, None] + torch.arange(args.ctx, device=dev)
        return arr[:, idx].permute(1, 2, 0, 3)

    assert args.eval_windows % args.batch == 0
    gen = torch.Generator(device=dev).manual_seed(1234)
    val_offs = torch.randint(0, len(val_data) - args.ctx - 1, (args.eval_windows,), device=dev, generator=gen)

    @torch.no_grad()
    def evaluate(frozen: bool = False) -> float:
        model.eval()
        tot = 0.0
        for k in range(0, args.eval_windows, args.batch):
            offs = val_offs[k : k + args.batch]
            x, y = make_batch(val_data, offs)
            cf = make_cf(cmem_frozen, 0, offs) if frozen else make_cf(cmem, 90_000_000, offs)
            tot += F.cross_entropy(forward_logits(x, cf).reshape(-1, 256), y.reshape(-1)).item()
        model.train()
        return tot / (args.eval_windows // args.batch) / math.log(2)

    out_path = args.out or os.path.join("checkpoints", f"concordance_{args.preset}.pt")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    start_it, seen, train_time = 0, 0, 0.0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["state"])
        for o, sd in zip(opt.opts, ck["opts"]):
            o.load_state_dict(sd)
        start_it, seen = ck["it"], ck.get("seen", 0)
        print(f"resumed from {args.resume} at step {start_it}", flush=True)

    for it in range(start_it + 1, args.iters + 1):
        t0 = time.time()
        frac = it / args.iters
        lr = (
            args.lr
            * min(1.0, it / max(1, args.warmup))
            * max(0.0, min(1.0, (1.0 - frac) / max(args.decay_frac, 1e-9)))
        )
        for grp in opt.param_groups:
            grp["lr"] = lr

        offs = torch.randint(0, len(train_data) - args.ctx - 1, (args.batch,), device=dev)
        x, y = make_batch(train_data, offs)
        cf = make_cf(cmem, 0, offs)

        logits = forward_logits(x, cf)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        seen += x.numel()
        if it > 5:
            torch.cuda.synchronize()
            train_time += time.time() - t0
        elif it == 5:
            torch.cuda.synchronize()

        if it % args.eval_every == 0 or it == args.iters:
            val_live = evaluate(frozen=False)
            extra = f"  val_frozen {evaluate(frozen=True):.3f}" if cfg["corpus_orders"] else ""
            ms = train_time / max(1, it - 5) * 1000.0
            print(
                f"it {it:6d}  train {loss.item() / math.log(2):.3f}  val_live {val_live:.3f}{extra} bpb  "
                f"bytes {seen / 1e6:.1f}M  [{train_time / 60:.1f}m, {ms:.0f} ms/step]",
                flush=True,
            )

        if args.save_every and it % args.save_every == 0 and it < args.iters:
            resume_path = out_path.replace(".pt", ".resume.pt")
            torch.save(
                {
                    "model": "hybrid",
                    "cfg": model.config(),
                    "state": model.state_dict(),
                    "it": it,
                    "seen": seen,
                    "opts": [o.state_dict() for o in opt.opts],
                },
                resume_path,
            )

    torch.save({"model": "hybrid", "cfg": model.config(), "state": model.state_dict()}, out_path)
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    print(f"saved {out_path}  (peak VRAM {peak_gb:.1f} GiB)", flush=True)


if __name__ == "__main__":
    main()
