"""On-demand disk-backed LiveMemory for streaming evaluation, generation, and zero-training corpus swaps.

Builds a sorted 64-bit FNV-1a polynomial-hash index per suffix order in low RAM (~200 MB peak).
At query time, binary search retrieves all corpus occurrences in chronological order, appends any
prompt / generated / test bytes streamed since `reset()`, and returns the exact same uint8 [orders, 5]
features (b1, Nq, r1, b2, r2) produced by `concordance.memory.build_corpus_memory`.
"""

from __future__ import annotations

import gc
import os
import time
from typing import Iterable, Sequence

import numpy as np

from .memory import ORDERS, quantize_count

_FNV_PRIME = np.uint64(1099511628211)
_MASK64 = (1 << 64) - 1


def hash_context(ctx: bytes | bytearray | Iterable[int]) -> np.uint64:
    """64-bit polynomial rolling hash of a byte suffix (oldest byte first)."""
    h = 0
    for c in ctx:
        h = (h * 1099511628211 + int(c) + 1) & _MASK64
    return np.uint64(h)


def build_live_index(
    src_path: str,
    out_dir: str,
    start: int = 0,
    end: int | None = 90_000_000,
    orders: Sequence[int] = ORDERS,
    chunk: int = 4_000_000,
    buckets: int = 32,
    verbose: bool = True,
) -> None:
    """Build a memory-mapped sorted hash index for bytes [start:end] of `src_path`."""
    os.makedirs(out_dir, exist_ok=True)
    raw = np.memmap(src_path, dtype=np.uint8, mode="r")
    x = raw[start : len(raw) if end is None else end]
    shift = np.uint64(64 - int(np.log2(buckets)))

    for n in orders:
        h_path = os.path.join(out_dir, f"h{n}.npy")
        nx_path = os.path.join(out_dir, f"nx{n}.npy")
        tmp_path = os.path.join(out_dir, f"tmp{n}.npy")
        if os.path.exists(nx_path) and os.path.exists(h_path) and not os.path.exists(tmp_path):
            continue

        t0 = time.time()
        M = len(x) - n
        if M <= 0:
            raise ValueError(f"Slice length {len(x)} is too short for order {n}")

        tmp = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.uint64, shape=(M,))
        for i in range(0, M, chunk):
            e = min(M, i + chunk)
            xs = np.asarray(x[i : e + n])
            h = np.zeros(e - i, dtype=np.uint64)
            for k in range(n):
                np.multiply(h, _FNV_PRIME, out=h)
                np.add(h, xs[k : k + e - i], out=h, casting="unsafe")
                np.add(h, np.uint64(1), out=h)
            tmp[i:e] = h
        tmp.flush()

        hs = np.lib.format.open_memmap(h_path, mode="w+", dtype=np.uint64, shape=(M,))
        ns = np.lib.format.open_memmap(nx_path, mode="w+", dtype=np.uint8, shape=(M,))
        at = 0
        for b in range(buckets):
            hv, iv = [], []
            for i in range(0, M, chunk):
                hc = np.asarray(tmp[i : i + chunk])
                sel = np.nonzero((hc >> shift) == b)[0]
                if len(sel):
                    hv.append(hc[sel])
                    iv.append(sel + i)
            if not hv:
                continue
            hv_arr = np.concatenate(hv)
            iv_arr = np.concatenate(iv)
            order = np.argsort(hv_arr, kind="stable")
            hs[at : at + len(order)] = hv_arr[order]
            ns[at : at + len(order)] = x[iv_arr[order] + n]
            at += len(order)
            del hv, iv, hv_arr, iv_arr, order

        assert at == M, f"indexed {at} != {M}"
        hs.flush()
        ns.flush()
        del hs, ns, tmp
        gc.collect()
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        if verbose:
            print(f"live index order {n:2d}: {time.time() - t0:.1f}s ({M:,} positions)", flush=True)


class LiveMemory:
    """Streaming exact-context memory over one or more disk indexes plus live text."""

    def __init__(self, path: str | Sequence[str] | None = None, orders: Sequence[int] = ORDERS):
        self.orders = tuple(orders)
        self.parts: list[tuple[dict[int, np.ndarray], dict[int, np.ndarray]]] = []
        if path is not None:
            paths = [path] if isinstance(path, str) else list(path)
            for p in paths:
                h_map = {n: np.load(os.path.join(p, f"h{n}.npy"), mmap_mode="r") for n in self.orders}
                nx_map = {n: np.load(os.path.join(p, f"nx{n}.npy"), mmap_mode="r") for n in self.orders}
                self.parts.append((h_map, nx_map))
        self.reset()

    def close(self) -> None:
        """Close underlying memory-mapped index handles (important on Windows before deleting files)."""
        for h_map, nx_map in self.parts:
            for arr in (*h_map.values(), *nx_map.values()):
                mm = getattr(arr, "_mmap", None)
                if mm is not None:
                    mm.close()
        self.parts.clear()

    def __enter__(self) -> "LiveMemory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def reset(self) -> None:
        self.text = bytearray()
        self.seen: dict[int, dict[bytes, bytearray]] = {n: {} for n in self.orders}

    def append(self, byte: int) -> None:
        """Append one byte; it becomes the continuation for every suffix ending just before it."""
        q = len(self.text)
        for n in self.orders:
            if q >= n:
                self.seen[n].setdefault(bytes(self.text[q - n : q]), bytearray()).append(byte)
        self.text.append(byte)

    def pop(self) -> None:
        """Undo the last `append` (used to roll back rejected speculative drafts)."""
        self.text.pop()
        q = len(self.text)
        for n in self.orders:
            if q >= n:
                key = bytes(self.text[q - n : q])
                buf = self.seen[n][key]
                buf.pop()
                if not buf:
                    del self.seen[n][key]

    def stats(self, n: int, ctx: bytes | bytearray, stream: bool = True) -> tuple[int, int, int, int, int]:
        """Return (N, b1, K1, b2, K2) for suffix `ctx` of length `n`."""
        hv = hash_context(ctx)
        segs = []
        for H, NX in self.parts:
            lo = int(np.searchsorted(H[n], hv, "left"))
            hi = int(np.searchsorted(H[n], hv, "right"))
            if hi > lo:
                segs.append(np.asarray(NX[n][lo:hi]))
        if stream:
            live_buf = self.seen[n].get(bytes(ctx))
            if live_buf:
                segs.append(np.frombuffer(bytes(live_buf), dtype=np.uint8))
        if not segs:
            return 0, 0, 0, 0, 0
        seq = segs[0] if len(segs) == 1 else np.concatenate(segs)
        N = len(seq)
        cnt = np.bincount(seq, minlength=256)
        b1 = int(seq[-1])
        other = np.nonzero(seq != b1)[0]
        if len(other):
            b2 = int(seq[other[-1]])
            k2 = int(cnt[b2])
        else:
            b2, k2 = 0, 0
        return N, b1, int(cnt[b1]), b2, k2

    def features(self, p: int | None = None, stream: bool = True) -> np.ndarray:
        """Return uint8 [len(orders), 5] features for the context ending at position p (default: last byte)."""
        p = len(self.text) - 1 if p is None else p
        out = np.zeros((len(self.orders), 5), dtype=np.uint8)
        for i, n in enumerate(self.orders):
            if p < n - 1:
                continue
            N, b1, K1, b2, K2 = self.stats(n, self.text[p - n + 1 : p + 1], stream=stream)
            if N == 0:
                continue
            Nq = quantize_count(N)
            r1 = int(round(255.0 * K1 / N))
            r2 = max(1, int(round(255.0 * K2 / N))) if K2 > 0 else 0
            out[i] = (b1, Nq, r1, b2, r2)
        return out
