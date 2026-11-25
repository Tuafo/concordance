"""Self-contained verification suite for Concordance (runs fast on CPU or CUDA).

Run with:
    python tests/test_concordance.py
    pytest tests/test_concordance.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concordance.live import LiveMemory, build_live_index
from concordance.memory import ORDERS, brute_force_stats, exact_stats, quantize_stats, suffix_classes
from concordance.model import Hybrid, RecallIndex, context_codes, preset_config
from concordance.muon import Muon, split_params


def _synthetic_bytes(seed: int = 0, length: int = 1200) -> bytes:
    rng = np.random.default_rng(seed)
    words = [bytes(rng.integers(97, 102, size=int(rng.integers(2, 7)))) for _ in range(25)]
    return b" ".join(words[int(i)] for i in rng.integers(0, len(words), size=400))[:length]


def test_exact_stats_matches_brute_force() -> None:
    xs = _synthetic_bytes(seed=0, length=900)
    x = torch.tensor(list(xs), dtype=torch.uint8)
    P, tr = len(xs), 600
    nx = torch.cat([x[1:].long(), torch.full((1,), 256, dtype=torch.long)])
    ar = torch.arange(P)
    seen_second = 0
    for n, c in suffix_classes(x, (4, 6, 12)):
        for mask, upto in ((ar < P - 1, P), (ar < tr - 1, tr)):
            got = [t.tolist() for t in exact_stats(c, nx, mask)]
            for p in range(n - 1, P):
                ref = brute_force_stats(xs, n, p, upto)
                mine = tuple(v[p] for v in got)
                seen_second += int(ref[4] > 0)
                assert mine == ref, f"order={n} p={p} upto={upto}: {mine} != {ref}"
    assert seen_second > 0


def test_live_memory_matches_exact_memory_and_pop() -> None:
    xs = _synthetic_bytes(seed=1, length=1200)
    tr = 800
    x = torch.tensor(list(xs), dtype=torch.uint8)
    P = len(xs)
    nx = torch.cat([x[1:].long(), torch.full((1,), 256, dtype=torch.long)])
    ar = torch.arange(P)

    ref_stream = np.zeros((len(ORDERS), P - tr, 5), dtype=np.uint8)
    ref_frozen = np.zeros_like(ref_stream)
    for i, (n, c) in enumerate(suffix_classes(x, ORDERS)):
        ref_stream[i] = quantize_stats(*exact_stats(c, nx, ar < P - 1))[tr:].numpy()
        ref_frozen[i] = quantize_stats(*exact_stats(c, nx, ar < tr - 1))[tr:].numpy()

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "corpus.bin")
        idx_dir = os.path.join(tmp, "live")
        with open(src, "wb") as fh:
            fh.write(xs)
        build_live_index(src, idx_dir, start=0, end=tr, orders=ORDERS, chunk=512, buckets=4, verbose=False)
        with LiveMemory(idx_dir, orders=ORDERS) as mem:
            # Feed context prefix strictly before `tr` (mirroring real streaming boundary)
            pre = max(ORDERS)
            for b in xs[tr - pre : tr]:
                mem.append(b)
            # Clear live additions from the pre-boundary bytes that are already in the disk index
            mem.seen = {n: {} for n in ORDERS}

            for k, b in enumerate(xs[tr:]):
                mem.append(b)
                # Test push/pop rollback invariance
                mem.append(120)
                mem.pop()
                got_frozen = mem.features(stream=False)
                got_stream = mem.features(stream=True)
                assert np.array_equal(got_frozen, ref_frozen[:, k]), f"frozen mismatch at {k}: {got_frozen} vs {ref_frozen[:, k]}"
                assert np.array_equal(got_stream, ref_stream[:, k]), f"stream mismatch at {k}: {got_stream} vs {ref_stream[:, k]}"


def test_recall_index_matches_quadratic() -> None:
    torch.manual_seed(2)
    B, P, d = 2, 48, 8
    x = torch.randint(0, 6, (B, P))
    v = torch.randn(B, P, d)
    causal = torch.ones(P, P, dtype=torch.bool).tril()
    for n in (0, 2, 4):
        idx = RecallIndex(x, n)
        mean, seen = idx.read(v)
        if n == 0:
            ref_m = torch.zeros_like(v)
            ref_s = torch.zeros(B, P, 1)
            for i in range(1, P):
                ref_m[:, i] = v[:, 1 : i + 1].mean(1)
                ref_s[:, i] = 1.0
        else:
            code = context_codes(x, n)
            wk = F.pad(code[:, :-1], (1, 0), value=-1)
            full_w = torch.arange(P) >= n
            m = ((code[:, :, None] == wk[:, None, :]) & causal & full_w[None, None, :]).float()
            cnt = m.sum(-1, keepdim=True)
            ref_m = (m @ v) / cnt.clamp(min=1.0)
            ref_s = cnt.clamp(max=1.0)
        assert torch.allclose(mean, ref_m, atol=1e-5), f"RecallIndex mean mismatch for n={n}"
        assert torch.allclose(seen, ref_s, atol=1e-6), f"RecallIndex seen mismatch for n={n}"


def test_sparse_attention_matches_brute_force() -> None:
    torch.manual_seed(0)
    B, H, P, hd, W, C, n = 2, 3, 256, 16, 64, 16, 4
    m = Hybrid(
        d=H * hd,
        layers=2,
        heads=H,
        ctx=P,
        window=W,
        mean_layers=(),
        recall_layers=(),
        corpus_orders=0,
        sparse_layers=(1,),
        chunk=C,
        topk=n,
    )
    q, k, v = (torch.randn(B, H, P, hd) for _ in range(3))
    fast = m._sparse(q, k, v)
    NC = P // C
    kc, vc = k.view(B, H, NC, C, hd).mean(3), v.view(B, H, NC, C, hd).mean(3)
    ref = torch.zeros_like(fast)
    for b in range(B):
        for h in range(H):
            for blk in range(P // W):
                far = [c for c in range(NC) if (c + 1) * C <= (blk - 1) * W]
                qm = q[b, h, (blk - 1) * W : blk * W].mean(0) if blk else torch.zeros(hd)
                sel = sorted(far, key=lambda c: -(qm @ kc[b, h, c]).item())[:n]
                for i in range(blk * W, (blk + 1) * W):
                    keys = (
                        [kc[b, h, c] for c in far]
                        + [k[b, h, c * C + t] for c in sel for t in range(C)]
                        + [k[b, h, j] for j in range(max(0, i - W + 1), i + 1)]
                    )
                    vals = (
                        [vc[b, h, c] for c in far]
                        + [v[b, h, c * C + t] for c in sel for t in range(C)]
                        + [v[b, h, j] for j in range(max(0, i - W + 1), i + 1)]
                    )
                    a = torch.softmax(torch.stack(keys) @ q[b, h, i] / hd**0.5, 0)
                    ref[b, h, i] = a @ torch.stack(vals)
    assert (fast - ref).abs().max().item() < 1e-5


def test_hybrid_causality_and_non_multiple_length() -> None:
    torch.manual_seed(3)
    for preset in ("w4k", "g4k", "s4k"):
        cfg = preset_config(preset)
        cfg.update(d=48, layers=4, heads=3, ctx=256, window=32, mean_layers=(0,), recall_layers=(1,), chunk=16, topk=2)
        if preset == "g4k":
            cfg["global_layers"] = (2,)
        elif preset == "s4k":
            cfg["sparse_layers"] = (2,)
        model = Hybrid(**cfg).eval()
        for p in model.parameters():
            if p.ndim == 2 and p.abs().sum() == 0:
                torch.nn.init.normal_(p, std=0.02)

        # Length 137 is deliberately not a multiple of window=32 (generation path!)
        P = 137
        x = torch.randint(0, 256, (2, P))
        cf = torch.randint(0, 256, (2, P, 6, 5), dtype=torch.uint8)
        x2, cf2 = x.clone(), cf.clone()
        x2[:, 90:] = torch.randint(0, 256, (2, P - 90))
        cf2[:, 90:] = torch.randint(0, 256, (2, P - 90, 6, 5), dtype=torch.uint8)
        with torch.no_grad():
            y1 = model(x, cf)
            y2 = model(x2, cf2)
        assert (y1[:, :90] - y2[:, :90]).abs().max().item() < 1e-5, f"causality breach in {preset}"
        assert (y1[:, 90:] - y2[:, 90:]).abs().max().item() > 1e-3


def test_muon_backward_step() -> None:
    torch.manual_seed(4)
    cfg = preset_config("g4k")
    cfg.update(d=48, layers=4, heads=3, ctx=128, window=32, mean_layers=(0,), recall_layers=(1,), global_layers=(2,))
    model = Hybrid(**cfg)
    mu, ad = split_params(model)
    assert len(mu) > 0 and len(ad) > 0
    opt_mu = Muon(mu, lr=1e-3, weight_decay=0.1, bf16=False)
    opt_ad = torch.optim.AdamW(ad, lr=1e-3)
    x = torch.randint(0, 256, (2, 65))
    cf = torch.randint(0, 256, (2, 64, 6, 5), dtype=torch.uint8)
    logits = model(x[:, :-1], cf)
    loss = F.cross_entropy(logits.reshape(-1, 256), x[:, 1:].reshape(-1))
    loss.backward()
    opt_mu.step()
    opt_ad.step()
    assert torch.isfinite(loss)


if __name__ == "__main__":
    for fn in (
        test_exact_stats_matches_brute_force,
        test_live_memory_matches_exact_memory_and_pop,
        test_recall_index_matches_quadratic,
        test_sparse_attention_matches_brute_force,
        test_hybrid_causality_and_non_multiple_length,
        test_muon_backward_step,
    ):
        fn()
        print(f"PASS  {fn.__name__}")
