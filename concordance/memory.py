"""Exact causal corpus memory over byte streams (prefix doubling on GPU/CPU).

For every position p and every suffix order n in ORDERS = (4, 6, 8, 12, 16, 24),
computes exact, collision-free statistics over all earlier writer positions j < p
sharing the same n-byte suffix x[j-n+1 .. j]:

    b1 : byte that followed the most recent earlier occurrence
    Nq : quantized log2 count of earlier occurrences, 1 + floor(4 * log2(N)) (0 if unseen)
    r1 : quantized share K1 / N for b1, in 0..255
    b2 : most recent earlier next-byte different from b1 (0 if none)
    r2 : quantized share K2 / N for b2, in 1..255 (0 if none)

Two evaluation modes:
    stream : every earlier byte in the stream (including prior test/val bytes) is a writer
    frozen : only training bytes (j < train_end - 1) are writers (static evaluation)
"""

from __future__ import annotations

import math
import time
from typing import Iterator, Sequence

import numpy as np
import torch

ORDERS: tuple[int, ...] = (4, 6, 8, 12, 16, 24)

_DOUBLING_PLAN: dict[int, tuple[int, int]] = {
    2: (1, 1),
    4: (2, 2),
    6: (4, 2),
    8: (4, 4),
    12: (8, 4),
    16: (8, 8),
    24: (16, 8),
    32: (16, 16),
    48: (32, 16),
    64: (32, 32),
    128: (64, 64),
    256: (128, 128),
}


def _shift(c: torch.Tensor, k: int) -> torch.Tensor:
    return torch.cat([torch.full((k,), -1, dtype=c.dtype, device=c.device), c[:-k]])


def _pair_classes(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Assign dense exact int32 equivalence-class IDs to pairs (a, b)."""
    al, bl = a.long(), b.long()
    key = al * (int(bl.max()) + 2) + (bl + 1)
    del al, bl
    return torch.unique(key, return_inverse=True)[1].int()


def _group_starts(sorted_keys: torch.Tensor, ar: torch.Tensor) -> torch.Tensor:
    gs = torch.ones_like(sorted_keys, dtype=torch.bool)
    gs[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return torch.cummax(torch.where(gs, ar, 0), 0).values


def suffix_classes(x: torch.Tensor, orders: Sequence[int] = ORDERS) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield (order, exact_class_ids) for each requested suffix length via prefix doubling."""
    c: dict[int, torch.Tensor] = {1: x.int()}
    need = set(orders)
    for n in sorted(need):
        stack = [n]
        while stack:
            m = stack.pop()
            if m in _DOUBLING_PLAN:
                for q in _DOUBLING_PLAN[m]:
                    if q not in need:
                        need.add(q)
                        stack.append(q)
    for n in sorted(need - {1}):
        a, b = _DOUBLING_PLAN[n]
        c[n] = _pair_classes(c[a], _shift(c[b], a))
        later = [m for m in need if m > n]
        for q in list(c):
            if q != n and not any(q in _DOUBLING_PLAN[m] for m in later):
                del c[q]
        if n in orders:
            yield n, c[n]


def exact_stats(c: torch.Tensor, nx: torch.Tensor, writers: torch.Tensor) -> list[torch.Tensor]:
    """Compute exact (N, b1, K1, b2, K2) per position for context classes `c` and writer mask `writers`."""
    P = c.numel()
    dev = c.device
    ar = torch.arange(P, device=dev)
    cl = c.long()
    order = torch.sort(cl, stable=True).indices
    start = _group_starts(cl[order], ar)
    ws = writers[order].long()

    ex = torch.cumsum(ws, 0) - ws
    N = ex - ex[start]
    del ex

    lw = torch.cummax(torch.where(ws.bool(), ar, -1), 0).values
    lw = torch.cat([lw.new_full((1,), -1), lw[:-1]])
    has = lw >= start

    # For each writer, find the most recent earlier writer in the same class with a different next byte.
    wi = torch.nonzero(ws, as_tuple=True)[0]
    nx_w, st_w = nx[order[wi]], start[wi]
    rs = torch.ones_like(wi, dtype=torch.bool)
    rs[1:] = (st_w[1:] != st_w[:-1]) | (nx_w[1:] != nx_w[:-1])
    ar_w = torch.arange(wi.numel(), device=dev)
    prev = torch.cummax(torch.where(rs, ar_w, 0), 0).values - 1
    ok = (prev >= 0) & (wi[prev.clamp(min=0)] >= st_w)
    pd = torch.full((P,), -1, dtype=torch.long, device=dev)
    pd[wi] = torch.where(ok, wi[prev.clamp(min=0)], -1)
    del wi, nx_w, st_w, rs, ar_w, prev, ok, start, ws

    j1 = lw.clamp(min=0)
    j2 = torch.where(has, pd[j1], -1)
    has2 = j2 >= 0
    del pd, lw

    src1, src2 = order[j1], order[j2.clamp(min=0)]
    b1 = torch.where(has, nx[src1], 0)
    b2 = torch.where(has2, nx[src2], 0)

    key2 = cl * 257 + nx
    order2 = torch.sort(key2, stable=True).indices
    start2 = _group_starts(key2[order2], ar)
    del key2
    ws2 = writers[order2].long()
    inc = torch.cumsum(ws2, 0)
    cnt_sorted = inc - (inc - ws2)[start2]
    del inc, ws2, start2

    cnt = torch.empty_like(cnt_sorted)
    cnt[order2] = cnt_sorted
    del order2, cnt_sorted

    K1 = torch.where(has, cnt[src1], 0)
    K2 = torch.where(has2, cnt[src2], 0)

    out = []
    for v in (N, b1, K1, b2, K2):
        o = torch.empty_like(v)
        o[order] = v
        out.append(o)
    return out


def quantize_count(n: int) -> int:
    """Scalar version of the uint8 log-count bucket (matches quantize_stats exactly)."""
    if n <= 0:
        return 0
    return min(255, 1 + int(math.floor(4.0 * math.log2(float(n)))))


def quantize_stats(
    N: torch.Tensor,
    b1: torch.Tensor,
    K1: torch.Tensor,
    b2: torch.Tensor,
    K2: torch.Tensor,
) -> torch.Tensor:
    """Pack exact counts into uint8 [P, 5] = (b1, Nq, r1, b2, r2)."""
    Nf = N.clamp(min=1).double()
    Nq = torch.where(N > 0, (1 + torch.floor(4.0 * torch.log2(Nf))).clamp(max=255), 0)
    r1 = torch.where(N > 0, torch.round(255.0 * K1.double() / Nf), 0)
    r2 = torch.where(K2 > 0, torch.round(255.0 * K2.double() / Nf).clamp(min=1), 0)
    return torch.stack([b1, Nq.long(), r1.long(), b2, r2.long()], -1).to(torch.uint8)


def build_corpus_memory(
    data_path: str,
    stream_out: str,
    val_frozen_out: str,
    train_end: int = 90_000_000,
    val_end: int = 95_000_000,
    orders: Sequence[int] = ORDERS,
    device: str = "cuda",
    verbose: bool = True,
) -> None:
    """Build streaming train+val memory [orders, val_end, 5] and frozen val memory [orders, val_end-train_end, 5]."""
    raw = np.fromfile(data_path, dtype=np.uint8, count=val_end)
    if len(raw) < val_end:
        raise ValueError(f"{data_path} has {len(raw):,} bytes, need at least {val_end:,}")
    x = torch.from_numpy(raw.copy()).to(device)
    P = x.numel()
    nx = torch.cat([x[1:].long(), torch.full((1,), 256, dtype=torch.long, device=device)])
    ar = torch.arange(P, device=device)

    mm = np.lib.format.open_memmap(stream_out, mode="w+", dtype=np.uint8, shape=(len(orders), P, 5))
    frozen = np.empty((len(orders), P - train_end, 5), dtype=np.uint8)
    t0 = time.time()
    for i, (n, c) in enumerate(suffix_classes(x, orders)):
        ti = time.time()
        mm[i] = quantize_stats(*exact_stats(c, nx, ar < P - 1)).cpu().numpy()
        frozen[i] = quantize_stats(*exact_stats(c, nx, ar < train_end - 1))[train_end:].cpu().numpy()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        if verbose:
            print(f"train/val order {n:2d}: {time.time() - ti:.1f}s", flush=True)
    mm.flush()
    del mm
    np.save(val_frozen_out, frozen)
    if verbose:
        print(f"wrote {stream_out} and {val_frozen_out} in {time.time() - t0:.0f}s", flush=True)


def build_test_memory(
    data_path: str,
    stream_out: str,
    frozen_out: str,
    train_end: int = 90_000_000,
    test_start: int = 95_000_000,
    test_end: int = 100_000_000,
    orders: Sequence[int] = ORDERS,
    device: str = "cuda",
    verbose: bool = True,
) -> None:
    """Build benchmark-clean official test memory (validation bytes train_end..test_start are NEVER writers)."""
    raw = np.fromfile(data_path, dtype=np.uint8, count=test_end)
    if len(raw) < test_end:
        raise ValueError(f"{data_path} has {len(raw):,} bytes, need at least {test_end:,}")
    x = torch.from_numpy(raw.copy()).to(device)
    P = x.numel()
    nx = torch.cat([x[1:].long(), torch.full((1,), 256, dtype=torch.long, device=device)])
    ar = torch.arange(P, device=device)

    w_frozen = ar < train_end - 1
    w_stream = w_frozen | ((ar >= test_start) & (ar < P - 1))
    stream = np.empty((len(orders), test_end - test_start, 5), dtype=np.uint8)
    frozen = np.empty_like(stream)

    t0 = time.time()
    for i, (n, c) in enumerate(suffix_classes(x, orders)):
        ti = time.time()
        stream[i] = quantize_stats(*exact_stats(c, nx, w_stream))[test_start:].cpu().numpy()
        frozen[i] = quantize_stats(*exact_stats(c, nx, w_frozen))[test_start:].cpu().numpy()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        if verbose:
            print(f"test order {n:2d}: {time.time() - ti:.1f}s", flush=True)
    np.save(stream_out, stream)
    np.save(frozen_out, frozen)
    if verbose:
        nxt = raw[test_start + 1 : test_end]
        for label, feat in (("stream", stream), ("frozen", frozen)):
            has = feat[2, :-1, 1] > 0
            acc = (feat[2, :-1, 0][has] == nxt[has]).mean() if has.any() else 0.0
            print(f"{label}: order-{orders[2]} coverage {has.mean():.1%}, top-1 accuracy {acc:.1%}", flush=True)
        print(f"wrote {stream_out} and {frozen_out} in {time.time() - t0:.0f}s", flush=True)


def brute_force_stats(xs: bytes, n: int, p: int, upto: int) -> tuple[int, int, int, int, int]:
    """Reference O(P) implementation for testing exact_stats at a single position p."""
    occ = [j for j in range(n - 1, p) if j + 1 < upto and xs[j - n + 1 : j + 1] == xs[p - n + 1 : p + 1]]
    if p < n - 1 or not occ:
        return 0, 0, 0, 0, 0
    b1 = xs[occ[-1] + 1]
    K1 = sum(xs[j + 1] == b1 for j in occ)
    other = [j for j in occ if xs[j + 1] != b1]
    if not other:
        return len(occ), b1, K1, 0, 0
    b2 = xs[other[-1] + 1]
    return len(occ), b1, K1, b2, sum(xs[j + 1] == b2 for j in occ)
