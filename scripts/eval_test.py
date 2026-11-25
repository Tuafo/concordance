"""Official enwik8 evaluation (every byte of bytes 95M-100M scored once, half-window stride).

Reports both:
    live   (stream) : training split + already-read test bytes in the index (comparable to dynamic evaluation)
    static (frozen) : training split only in the index (validation split is NEVER indexed)

Works with checkpoints trained by `scripts/train.py` as well as archived `text_hybrid_*.pt` checkpoints.

Example:
    python scripts/eval_test.py checkpoints/concordance_g4k.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concordance.model import Hybrid


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a Concordance checkpoint on the official enwik8 test/val split.")
    ap.add_argument("ckpt", help="Path to trained checkpoint (.pt)")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--data", default="data/enwik8")
    ap.add_argument("--mem-dir", default="data")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = args.device
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if ck.get("model") != "hybrid":
        raise ValueError(f"Expected a 'hybrid' checkpoint, got model={ck.get('model')!r}")

    cfg = ck["cfg"]
    model = Hybrid(**cfg).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()

    ctx = cfg.get("ctx", 4096)
    lo = 95_000_000 if args.split == "test" else 90_000_000
    raw = np.fromfile(args.data, dtype=np.uint8, count=lo + 5_000_000)
    data = torch.from_numpy(raw[lo:].copy()).to(dev)
    N = data.numel()

    if args.split == "test":
        modes = {
            "live": os.path.join(args.mem_dir, "corpus_mem2_test.npy"),
            "static": os.path.join(args.mem_dir, "corpus_mem2_test_frozen.npy"),
        }
    else:
        modes = {
            "live": None,
            "static": os.path.join(args.mem_dir, "corpus_mem2_val_frozen.npy"),
        }

    stride = ctx // 2
    starts = list(range(0, N - ctx, stride))
    if starts[-1] + ctx < N - 1:
        starts.append(N - 1 - ctx)

    use_bf16 = dev.startswith("cuda")
    for mode, path in modes.items():
        mem = None
        if cfg.get("corpus_orders"):
            if path is None:
                arr = np.load(os.path.join(args.mem_dir, "corpus_mem2.npy"), mmap_mode="r")[:, lo : lo + N]
            else:
                arr = np.load(path, mmap_mode="r")
            mem = torch.from_numpy(np.array(arr, copy=True)).to(dev)

        tot, cnt, prev_end = 0.0, 0, 0
        with torch.no_grad(), torch.autocast(device_type="cuda" if use_bf16 else "cpu", dtype=torch.bfloat16, enabled=use_bf16):
            for b in range(0, len(starts), args.batch):
                st = starts[b : b + args.batch]
                idx = torch.tensor(st, device=dev)[:, None] + torch.arange(ctx + 1, device=dev)
                w = data[idx].long()
                cf = mem[:, idx[:, :-1]].permute(1, 2, 0, 3) if mem is not None else None
                logits = (model(w[:, :-1], cf) if cf is not None else model(w[:, :-1])).float()
                loss = F.cross_entropy(logits.reshape(-1, 256), w[:, 1:].reshape(-1), reduction="none").view(len(st), ctx)
                for r, s in enumerate(st):
                    a0 = max(prev_end, s) - s
                    tot += loss[r, a0:].sum().item()
                    cnt += ctx - a0
                    prev_end = s + ctx
        bpb = tot / cnt / math.log(2)
        print(f"{args.ckpt}  [{args.split}, {mode:6s}]: {bpb:.4f} bpb  ({cnt:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
