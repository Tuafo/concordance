"""Concordance hybrid reader: CorpusMem + RecallBlock + Hybrid byte-level LM.

Preserves exact state_dict compatibility with trained checkpoints from the enwik8-era runs
(`g4k`, `w4k`, `s4k`, `new_d0`) while supporting arbitrary sequence lengths during generation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 256


def context_codes(x: torch.Tensor, n: int) -> torch.Tensor:
    """Exact int64 code of the n bytes ending at each position (polynomial wrap mod 2^64 for n > 7)."""
    xp = F.pad(x.long(), (n - 1, 0), value=PAD)
    code = torch.zeros_like(x, dtype=torch.long)
    if n > 7:
        for k in range(n):
            code = code * 257 + xp[:, k : xp.shape[1] - (n - 1 - k)]
        return code
    for k in range(n):
        code = code + xp[:, n - 1 - k : xp.shape[1] - k] * (257 ** k)
    return code


class RecallIndex:
    """O(P log P) exact in-window context-recall index over a sequence batch."""

    def __init__(self, x: torch.Tensor, n: int, code: torch.Tensor | None = None):
        B, P = (x if code is None else code).shape
        dev = (x if code is None else code).device
        self.glob = n == 0
        if self.glob:
            self.count = torch.arange(P, device=dev, dtype=torch.float32).expand(B, 1, P)
            return
        pos = torch.arange(P, device=dev)
        if code is None:
            code = context_codes(x, n)
        wk = F.pad(code[:, :-1], (1, 0), value=-1)
        wk = torch.where(pos >= n, wk, -1 - pos)
        keys = torch.stack([wk, code], -1).view(B, 2 * P)
        perm = torch.sort(keys, dim=1, stable=True).indices
        skey = keys.gather(1, perm)
        start = torch.ones_like(skey, dtype=torch.bool)
        start[:, 1:] = skey[:, 1:] != skey[:, :-1]
        ar = torch.arange(2 * P, device=dev).expand(B, -1)
        start_idx = torch.cummax(torch.where(start, ar, 0), 1).values
        writer = perm % 2 == 0
        inv = torch.empty_like(perm).scatter_(1, perm, ar)
        reader_at = inv[:, 1::2]
        nw = writer.long().cumsum(1)
        self.wsrc = (perm // 2).gather(1, torch.sort((~writer).to(torch.int8), dim=1, stable=True).indices[:, :P])
        self.r_end = nw.gather(1, reader_at)
        self.r_base = (nw - writer.long()).gather(1, start_idx.gather(1, reader_at))
        self.count = (self.r_end - self.r_base).to(torch.float32)[:, None]

    def running(self, ev: torch.Tensor) -> torch.Tensor:
        if self.glob:
            return ev.cumsum(-1)
        d = ev.shape[1]
        cs = F.pad(ev.double().cumsum(-1), (1, 0))
        end = cs.gather(2, self.r_end[:, None].expand(-1, d, -1))
        base = cs.gather(2, self.r_base[:, None].expand(-1, d, -1))
        return (end - base).to(ev.dtype)

    def writers(self, v: torch.Tensor) -> torch.Tensor:
        vT = v.transpose(1, 2)
        if self.glob:
            return F.pad(vT[..., 1:], (1, 0))
        return vT.gather(2, self.wsrc[:, None].expand(-1, vT.shape[1], -1))

    def read(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.running(self.writers(v)) / self.count.clamp(min=1.0)
        return mean.transpose(1, 2), self.count.clamp(max=1.0).transpose(1, 2)

    def read_ll(self, v: torch.Tensor, k: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Local-linear group read: mean(v) + p . Cov(k, v)."""
        B, P, dv = v.shape
        dk = k.shape[-1]
        kv = (k[..., :, None] * v[..., None, :]).reshape(B, P, dk * dv)
        S = self.running(self.writers(torch.cat([kv, k, v], -1))).transpose(1, 2)
        n = self.count.transpose(1, 2).clamp(min=1.0)
        mk = S[..., dk * dv : dk * dv + dk] / n
        mv = S[..., dk * dv + dk :] / n
        cov = S[..., : dk * dv].reshape(B, P, dk, dv) / n[..., None] - mk[..., :, None] * mv[..., None, :]
        return mv + torch.einsum("bpk,bpkv->bpv", p, cov), self.count.clamp(max=1.0).transpose(1, 2)


class RecallBlock(nn.Module):
    """Exact-context recall layer (replaces attention; O(P) running-sum read over exact byte groups)."""

    def __init__(
        self,
        d: int,
        orders: tuple[int, ...] = (0, 2, 4, 6),
        dv: int = 32,
        dk: int = 8,
        query_orders: tuple[int, ...] = (0, 2),
    ):
        super().__init__()
        self.orders, self.dv, self.dk = tuple(orders), dv, dk
        self.kq = [2 * dk if n in query_orders else 0 for n in self.orders]
        self.write = nn.Linear(d, len(self.orders) * dv + sum(self.kq))
        with torch.no_grad():
            a = 0
            for kq in self.kq:
                a += dv
                if kq:
                    self.write.weight[a : a + kq // 2].mul_(0.5)
                    self.write.bias[a : a + kq // 2].zero_()
                    self.write.weight[a + kq // 2 : a + kq].zero_()
                    self.write.bias[a + kq // 2 : a + kq].zero_()
                a += kq
        self.out = nn.Linear(len(self.orders) * (dv + 1), d)

    def macs(self, d: int) -> int:
        reads = sum(
            ((self.dk + 1) * (self.dv + 1) * 4 + 2 * self.dk * self.dv) if kq else self.dv * 4
            for kq in self.kq
        )
        return d * self.write.out_features + self.out.in_features * d + reads

    def forward(self, h: torch.Tensor, M: list[RecallIndex]) -> torch.Tensor:
        dv, dk = self.dv, self.dk
        vals = self.write(h).split([dv + kq for kq in self.kq], -1)
        reads = []
        for idx, v, kq in zip(M, vals, self.kq):
            if kq:
                r = idx.read_ll(v[..., :dv], torch.tanh(v[..., dv : dv + dk]), v[..., dv + dk :])
            else:
                r = idx.read(v[..., :dv])
            reads.extend(r)
        return self.out(torch.cat(reads, -1))


class CorpusMem(nn.Module):
    """Input embedding projection and zero-initialized gated logit copy path for corpus memory features."""

    HIT_WINDOWS = (8, 64)

    def __init__(
        self,
        R: int,
        emb_dim: int,
        C: int,
        bdim: int = 16,
        hits: bool = False,
        cands: int = 1,
        distinct: bool = False,
    ):
        super().__init__()
        self.R, self.hits, self.cands, self.distinct = R, hits, cands, distinct
        n_sc = 3 + 2 * (cands - 1) + (1 + len(self.HIT_WINDOWS) if hits else 0) + (2 if distinct else 0)
        self.byte = nn.Embedding(256, bdim)
        self.proj = nn.Linear(R * (cands * bdim + n_sc), emb_dim)
        self.gate = nn.Linear(C + n_sc * R, R * cands)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def parts(self, cf: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = cf[..., 0].long()
        has = (cf[..., 1] > 0).float()
        sc = [cf[..., 1].float() / 32.0, cf[..., 2].float() / 255.0, has]
        bs, hs = [b], [has]
        if self.cands == 2:
            has2 = (cf[..., 4] > 0).float()
            sc += [cf[..., 4].float() / 255.0, has2]
            bs.append(cf[..., 3].long())
            hs.append(has2)
        if self.distinct:
            sc += [torch.log1p(cf[..., 5].float()) / 5.55, cf[..., 6].float() / 255.0]
        if self.hits:
            prev_b = F.pad(b[:, :-1], (0, 0, 1, 0))
            prev_has = F.pad(has[:, :-1], (0, 0, 1, 0))
            hit = (prev_b == x[..., None]).float() * prev_has
            sc.append(hit)
            P = x.shape[1]
            ch = hit.transpose(1, 2).contiguous().cumsum(-1)
            cn = prev_has.transpose(1, 2).contiguous().cumsum(-1)
            for w in self.HIT_WINDOWS:
                hw = ch - F.pad(ch, (w, 0))[..., :P]
                nw = cn - F.pad(cn, (w, 0))[..., :P]
                sc.append(((hw + 0.5) / (nw + 1.0)).transpose(1, 2))
        return torch.stack(bs, -1), torch.stack(hs, -1), torch.stack(sc, -1)

    def embed(self, cf: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        b, has, sc = self.parts(cf, x)
        v = torch.cat([(self.byte(b) * has[..., None]).flatten(-2), sc], -1)
        return self.proj(v.flatten(-2))

    def copy(self, s: torch.Tensor, cf: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        b, has, sc = self.parts(cf, x)
        g = self.gate(torch.cat([s, sc.flatten(-2)], -1)) * has.flatten(-2)
        return torch.zeros(*s.shape[:-1], 256, device=s.device, dtype=s.dtype).scatter_add(-1, b.flatten(-2), g)


class Norm(nn.Module):
    def __init__(self, d: int, rms: bool = True):
        super().__init__()
        self.rms = rms
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = None if rms else nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rms:
            return F.rms_norm(x, (x.shape[-1],), self.weight)
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias)


def rope_tables(P: int, hd: int, device: torch.device, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / base ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd)
    f = torch.arange(P, device=device, dtype=torch.float32)[:, None] * inv[None]
    return f.cos(), f.sin()


def apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    a, b = t.float().chunk(2, -1)
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1).to(t.dtype)


class Block(nn.Module):
    def __init__(
        self,
        kind: str,
        d: int,
        heads: int,
        rms: bool = True,
        bias: bool = False,
        qk_norm: bool = True,
        relu2: bool = True,
        zero_init: bool = True,
        recall_orders: tuple[int, ...] = (0, 2, 4, 6),
    ):
        super().__init__()
        self.kind, self.H, self.qk_norm, self.relu2 = kind, heads, qk_norm, relu2
        self.norm1, self.norm2 = Norm(d, rms), Norm(d, rms)
        if kind == "attn":
            self.wqkv = nn.Linear(d, 3 * d, bias=bias)
            self.wo = nn.Linear(d, d, bias=bias)
        elif kind == "mean":
            self.wv = nn.Linear(d, d, bias=bias)
            self.wo = nn.Linear(d, d, bias=bias)
        else:
            self.recall = RecallBlock(d, recall_orders)
        self.w_up = nn.Linear(d, 4 * d, bias=bias)
        self.w_down = nn.Linear(4 * d, d, bias=bias)
        if zero_init:
            nn.init.zeros_(self.w_down.weight)
            if kind != "recall":
                nn.init.zeros_(self.wo.weight)

    def mlp(self, h: torch.Tensor) -> torch.Tensor:
        u = self.w_up(self.norm2(h))
        return self.w_down(F.relu(u).square() if self.relu2 else F.gelu(u))


class Hybrid(nn.Module):
    """Modern byte-level hybrid reader: running mean, sliding-window attention, exact recall, optional global/sparse layers."""

    def __init__(
        self,
        d: int = 384,
        layers: int = 8,
        heads: int = 6,
        ctx: int = 4096,
        window: int = 128,
        mean_layers: tuple[int, ...] = (0,),
        recall_layers: tuple[int, ...] = (4,),
        recall_orders: tuple[int, ...] = (0, 2, 4, 6),
        corpus_orders: int = 6,
        corpus_hits: bool = True,
        corpus_cands: int = 2,
        dropout: float = 0.0,
        rms: bool = True,
        bias: bool = False,
        rope: bool = True,
        qk_norm: bool = True,
        relu2: bool = True,
        zero_init: bool = True,
        softcap: float = 30.0,
        abs_pos: bool = False,
        global_layers: tuple[int, ...] = (3, 6),
        sparse_layers: tuple[int, ...] = (),
        chunk: int = 32,
        topk: int = 8,
    ):
        super().__init__()
        self.cfg = dict(
            d=d,
            layers=layers,
            heads=heads,
            ctx=ctx,
            window=window,
            mean_layers=tuple(mean_layers),
            recall_layers=tuple(recall_layers),
            recall_orders=tuple(recall_orders),
            corpus_orders=corpus_orders,
            corpus_hits=corpus_hits,
            corpus_cands=corpus_cands,
            dropout=dropout,
            rms=rms,
            bias=bias,
            rope=rope,
            qk_norm=qk_norm,
            relu2=relu2,
            zero_init=zero_init,
            softcap=softcap,
            abs_pos=abs_pos,
            global_layers=tuple(global_layers),
            sparse_layers=tuple(sparse_layers),
            chunk=chunk,
            topk=topk,
        )
        self.d, self.layers, self.ctx, self.window = d, layers, ctx, window
        self.mean_layers = tuple(mean_layers)
        self.recall_layers = tuple(recall_layers)
        self.recall_orders = tuple(recall_orders)
        self.global_layers = tuple(global_layers)
        self.sparse_layers = tuple(sparse_layers)
        self.chunk, self.topk = chunk, topk
        self.corpus_orders, self.dropout, self.rope, self.softcap = corpus_orders, dropout, rope, softcap

        self.embed = nn.Embedding(256, d)
        self.pos = nn.Parameter(torch.zeros(ctx, d)) if abs_pos else None
        if corpus_orders:
            self.corpus = CorpusMem(corpus_orders, d, d, hits=corpus_hits, cands=corpus_cands)
        kinds = [
            "mean" if i in self.mean_layers else "recall" if i in self.recall_layers else "attn"
            for i in range(layers)
        ]
        self.blocks = nn.ModuleList(
            Block(k, d, heads, rms, bias, qk_norm, relu2, zero_init, self.recall_orders) for k in kinds
        )
        self.norm = Norm(d, rms)
        self.head = nn.Linear(d, 256, bias=bias)
        if zero_init:
            nn.init.zeros_(self.head.weight)
        self.flex = False
        self._bm: dict[int, object] = {}

    def config(self) -> dict:
        return dict(self.cfg)

    def macs_per_byte(self, P: int) -> float:
        d = self.d
        keys = min(self.window, P) if self.window else P / 2
        attn = 12 * d ** 2 + 2 * d * keys
        mean = 10 * d ** 2 + 2 * d
        rec = sum(b.recall.macs(d) + 8 * d ** 2 for b in self.blocks if b.kind == "recall")
        n_attn = sum(b.kind == "attn" for b in self.blocks) - len(self.global_layers) - len(self.sparse_layers)
        far = len(self.global_layers) * (12 * d ** 2 + 2 * d * P / 2) + len(self.sparse_layers) * (
            12 * d ** 2
            + 2 * d * (P / 2 / self.chunk + self.topk * self.chunk + 2 * self.window)
            + 2 * d * P / self.chunk / 2
        )
        n_mean = sum(b.kind == "mean" for b in self.blocks)
        corpus = (
            self.corpus.proj.in_features * self.corpus.proj.out_features
            + self.corpus.gate.in_features * self.corpus.gate.out_features
            if self.corpus_orders
            else 0
        )
        return corpus + n_attn * attn + n_mean * mean + rec + far + d * 256

    def _block_mask(self, P: int, device: torch.device):
        if P not in self._bm:
            from torch.nn.attention.flex_attention import create_block_mask

            W = self.window
            self._bm[P] = create_block_mask(
                lambda b, h, qi, ki: (qi >= ki) & (qi - ki < W),
                B=None,
                H=None,
                Q_LEN=P,
                KV_LEN=P,
                device=device,
            )
        return self._bm[P]

    def _local(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exact causal sliding-window attention (works for any sequence length P)."""
        Bq, H, P, hd = q.shape
        W = self.window
        pad = (-P) % W
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
        Pp = P + pad
        nb = Pp // W
        qb = q.reshape(Bq * H, nb, W, hd)

        def two(t: torch.Tensor) -> torch.Tensor:
            tb = t.reshape(Bq * H, nb, W, hd)
            return torch.cat([F.pad(tb[:, :-1], (0, 0, 0, 0, 1, 0)), tb], 2)

        i = torch.arange(W, device=q.device)[:, None]
        j = torch.arange(2 * W, device=q.device)[None, :] - W
        allowed = (j <= i) & (j > i - W)
        mask = torch.cat([(allowed & (j >= 0))[None], allowed[None].expand(nb - 1, -1, -1)], 0)
        out = F.scaled_dot_product_attention(qb, two(k), two(v), attn_mask=mask).reshape(Bq, H, Pp, hd)
        return out[:, :, :P, :]

    def _sparse(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Causal chunk-selected far attention + exact local window (works for any length P)."""
        B, H, P, hd = q.shape
        W, C, n = self.window, self.chunk, self.topk
        assert W % C == 0, (W, C)
        pad = (-P) % W
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
        Pp = P + pad
        NB, NC = Pp // W, Pp // C
        kc = k.view(B, H, NC, C, hd).mean(3)
        vc = v.view(B, H, NC, C, hd).mean(3)
        far = (torch.arange(NC, device=q.device)[None, :] + 1) * C <= (torch.arange(NB, device=q.device)[:, None] - 1) * W
        n = min(n, NC)
        with torch.no_grad():
            qb = F.pad(q.view(B, H, NB, W, hd).mean(3)[:, :, :-1], (0, 0, 1, 0))
            s = (qb @ kc.transpose(-1, -2)).masked_fill(~far, float("-inf"))
            top, sel = s.topk(n, -1)
            ok = top > float("-inf")
            idx = (sel[..., None] * C + torch.arange(C, device=q.device)).reshape(B, H, NB * n * C)
        ks = k.gather(2, idx[..., None].expand(-1, -1, -1, hd)).view(B, H, NB, n * C, hd)
        vs = v.gather(2, idx[..., None].expand(-1, -1, -1, hd)).view(B, H, NB, n * C, hd)

        def two(t: torch.Tensor) -> torch.Tensor:
            tb = t.view(B, H, NB, W, hd)
            return torch.cat([F.pad(tb[:, :, :-1], (0, 0, 0, 0, 1, 0)), tb], 3)

        i = torch.arange(W, device=q.device)[:, None]
        j = torch.arange(2 * W, device=q.device)[None, :] - W
        allowed = (j <= i) & (j > i - W)
        local = torch.cat([(allowed & (j >= 0))[None], allowed[None].expand(NB - 1, -1, -1)], 0)
        K = torch.cat([kc[:, :, None].expand(-1, -1, NB, -1, -1), ks, two(k)], 3)
        V = torch.cat([vc[:, :, None].expand(-1, -1, NB, -1, -1), vs, two(v)], 3)
        mask = torch.cat(
            [
                far[None, None, :, None, :].expand(B, H, -1, W, -1),
                ok.repeat_interleave(C, -1)[:, :, :, None, :].expand(-1, -1, -1, W, -1),
                local[None, None].expand(B, H, -1, -1, -1),
            ],
            -1,
        )
        out = F.scaled_dot_product_attention(q.view(B, H, NB, W, hd), K, V, attn_mask=mask).view(B, H, Pp, hd)
        return out[:, :, :P, :]

    def attend(
        self,
        blk: Block,
        h: torch.Tensor,
        rope_cs: tuple[torch.Tensor, torch.Tensor] | None,
        li: int = -1,
    ) -> torch.Tensor:
        B, P, d = h.shape
        H = blk.H
        q, k, v = blk.wqkv(blk.norm1(h)).view(B, P, 3, H, d // H).permute(2, 0, 3, 1, 4)
        if blk.qk_norm:
            q, k = F.rms_norm(q, (q.shape[-1],)), F.rms_norm(k, (k.shape[-1],))
        if rope_cs is not None:
            q, k = apply_rope(q, *rope_cs), apply_rope(k, *rope_cs)
        if li in self.sparse_layers:
            o = self._sparse(q, k, v)
        elif li in self.global_layers:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif self.window and self.window < P:
            if self.flex and P in self._bm:
                from torch.nn.attention.flex_attention import flex_attention

                o = flex_attention(q, k, v, block_mask=self._block_mask(P, h.device))
            else:
                o = self._local(q, k, v)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return blk.wo(o.transpose(1, 2).reshape(B, P, d))

    def forward(self, x: torch.Tensor, cf: torch.Tensor | None = None) -> torch.Tensor:
        B, P = x.shape
        dr = (lambda t: F.dropout(t, self.dropout, True)) if (self.dropout and self.training) else (lambda t: t)
        h = self.embed(x)
        if self.pos is not None:
            h = h + self.pos[:P]
        if self.corpus_orders:
            if cf is None:
                raise ValueError("Model expects corpus memory features `cf`")
            h = h + self.corpus.embed(cf, x)
        h = dr(h)
        M = [RecallIndex(x, n) for n in self.recall_orders] if self.recall_layers else None
        rope_cs = rope_tables(P, self.d // self.blocks[0].H, x.device) if self.rope else None
        for li, blk in enumerate(self.blocks):
            if blk.kind == "attn":
                o = self.attend(blk, h, rope_cs, li)
            elif blk.kind == "mean":
                v = blk.wv(blk.norm1(h))
                cnt = torch.arange(1, P + 1, device=x.device, dtype=torch.float32)[:, None]
                o = blk.wo((torch.cumsum(v.float(), 1) / cnt).to(v.dtype))
            else:
                with torch.autocast(device_type=x.device.type, enabled=False):
                    rb = blk.recall
                    o = rb(blk.norm1(h).to(rb.write.weight.dtype), M).to(h.dtype)
            h = h + dr(o)
            h = h + dr(blk.mlp(h))
        h = self.norm(h)
        out = self.head(h)
        if self.corpus_orders:
            out = out + self.corpus.copy(h, cf, x)
        if self.softcap:
            out = self.softcap * torch.tanh(out.float() / self.softcap)
        return out


def preset_config(name: str) -> dict:
    """Return exact architecture hyperparameters for the published Concordance presets."""
    base = dict(
        d=384,
        layers=8,
        heads=6,
        ctx=4096,
        window=128,
        mean_layers=(0,),
        recall_layers=(4,),
        recall_orders=(0, 2, 4, 6),
        corpus_orders=6,
        corpus_hits=True,
        corpus_cands=2,
        dropout=0.0,
        rms=True,
        bias=False,
        rope=True,
        qk_norm=True,
        relu2=True,
        zero_init=True,
        softcap=30.0,
        abs_pos=False,
        global_layers=(),
        sparse_layers=(),
        chunk=32,
        topk=8,
    )
    if name == "g4k":
        base["global_layers"] = (3, 6)
    elif name == "w4k":
        pass
    elif name == "s4k":
        base["sparse_layers"] = (3, 6)
    else:
        raise ValueError(f"Unknown preset {name!r}; expected one of 'g4k', 'w4k', 's4k'")
    return base
