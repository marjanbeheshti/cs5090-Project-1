#!/usr/bin/env python3

"""

LLaMA-style decoder-only transformer.


Three differences from the original GPT-2 design, all standard since LLaMA:

  RMSNorm  instead of LayerNorm  (no mean subtraction, no bias -- cheaper)

  RoPE     instead of learned position embeddings (rotates q/k by position)

  SwiGLU   instead of GELU MLP   (gated activation, 3 matrices not 2)


Every component is a separate class so the ablation study can remove one

without touching anything else.

"""


import math

from dataclasses import dataclass


import torch

import torch.nn as nn

import torch.nn.functional as F



@dataclass

class ModelConfig:

    vocab_size: int = 8192

    dim: int = 384

    n_layers: int = 6

    n_heads: int = 6

    seq_len: int = 512

    dropout: float = 0.0

    norm_eps: float = 1e-5

    # Ablation switches. Defaults reproduce the baseline.

    use_rope: bool = True

    use_rmsnorm: bool = True
    use_pos: bool = True



class RMSNorm(nn.Module):

    """Scales each vector to unit root-mean-square, then applies a learned gain.


    Unlike LayerNorm it does not subtract the mean, which saves a pass over the

    data and in practice works just as well.

    """


    def __init__(self, dim: int, eps: float):

        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.ones(dim))


    def forward(self, x):

        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)

        return (x.float() * rms).type_as(x) * self.weight



def precompute_freqs_cis(head_dim: int, end: int, theta: float = 10000.0):

    """Rotation angles for RoPE, one complex number per (position, dim-pair)."""

    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

    t = torch.arange(end, dtype=torch.float32)

    freqs = torch.outer(t, freqs)

    return torch.polar(torch.ones_like(freqs), freqs)



def apply_rope(xq, xk, freqs_cis):

    """Rotate query and key vectors by their absolute position.


    Because a rotation by m then a dot product with a rotation by n depends only

    on (m - n), attention scores end up encoding RELATIVE distance for free.

    """

    # (B, T, H, D) -> complex pairs (B, T, H, D/2)

    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))

    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    f = freqs_cis[: xq.shape[1]].unsqueeze(0).unsqueeze(2)  # (1, T, 1, D/2)

    xq_out = torch.view_as_real(xq_c * f).flatten(3)

    xk_out = torch.view_as_real(xk_c * f).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)



class Attention(nn.Module):

    def __init__(self, cfg: ModelConfig):

        super().__init__()

        assert cfg.dim % cfg.n_heads == 0

        self.n_heads = cfg.n_heads

        self.head_dim = cfg.dim // cfg.n_heads

        self.use_rope = cfg.use_rope

        self.dropout = cfg.dropout


        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)

        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)

        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)

        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)

        self.resid_drop = nn.Dropout(cfg.dropout)


    def forward(self, x, freqs_cis):

        B, T, C = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)

        k = self.wk(x).view(B, T, self.n_heads, self.head_dim)

        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)


        if self.use_rope:

            q, k = apply_rope(q, k, freqs_cis)


        # (B, T, H, D) -> (B, H, T, D) for scaled_dot_product_attention

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))


        # is_causal masks the future; PyTorch picks a fused kernel when it can.

        out = F.scaled_dot_product_attention(

            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True

        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_drop(self.wo(out))



class FeedForward(nn.Module):

    """SwiGLU: one branch is gated by a SiLU-activated copy of the other.


    Hidden size is 8/3 * dim rather than 4 * dim, so the three matrices cost

    about the same as GPT-2's two.

    """


    def __init__(self, cfg: ModelConfig, multiple_of: int = 32):

        super().__init__()

        hidden = int(8 * cfg.dim / 3)

        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(cfg.dim, hidden, bias=False)  # gate

        self.w3 = nn.Linear(cfg.dim, hidden, bias=False)  # value

        self.w2 = nn.Linear(hidden, cfg.dim, bias=False)  # project back

        self.drop = nn.Dropout(cfg.dropout)


    def forward(self, x):

        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))



class Block(nn.Module):

    """Pre-norm residual block: x + attn(norm(x)), then x + ffn(norm(x))."""


    def __init__(self, cfg: ModelConfig):

        super().__init__()

        Norm = RMSNorm if cfg.use_rmsnorm else (lambda d, eps: nn.LayerNorm(d, eps=eps))

        self.attn_norm = Norm(cfg.dim, cfg.norm_eps)

        self.attn = Attention(cfg)

        self.ffn_norm = Norm(cfg.dim, cfg.norm_eps)

        self.ffn = FeedForward(cfg)


    def forward(self, x, freqs_cis):

        x = x + self.attn(self.attn_norm(x), freqs_cis)

        return x + self.ffn(self.ffn_norm(x))



class Transformer(nn.Module):

    def __init__(self, cfg: ModelConfig):

        super().__init__()

        self.cfg = cfg
        if not cfg.use_pos:
            cfg.use_rope = False

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)

        self.drop = nn.Dropout(cfg.dropout)

        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))

        Norm = RMSNorm if cfg.use_rmsnorm else (lambda d, eps: nn.LayerNorm(d, eps=eps))

        self.norm = Norm(cfg.dim, cfg.norm_eps)

        # Untied output head: separate matrix from the input embedding.

        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)


        # Learned absolute positions, used only when RoPE is ablated away.

        self.pos_emb = (

            None if (cfg.use_rope or not cfg.use_pos) else nn.Embedding(cfg.seq_len, cfg.dim)

        )


        self.register_buffer(

            "freqs_cis",

            precompute_freqs_cis(cfg.dim // cfg.n_heads, cfg.seq_len),

            persistent=False,

        )


        self.apply(self._init_weights)

        # Scale residual-path projections by depth (GPT-2 trick, keeps the

        # residual stream's variance stable as layers stack up).

        for name, p in self.named_parameters():

            if name.endswith("w2.weight") or name.endswith("wo.weight"):

                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))


    def _init_weights(self, module):

        if isinstance(module, nn.Linear):

            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def num_params(self, non_embedding: bool = False) -> int:

        n = sum(p.numel() for p in self.parameters())

        if non_embedding:

            n -= self.tok_emb.weight.numel() + self.output.weight.numel()

        return n


    def forward(self, idx, targets=None):

        B, T = idx.shape

        assert T <= self.cfg.seq_len, f"sequence {T} exceeds {self.cfg.seq_len}"


        x = self.tok_emb(idx)

        if self.pos_emb is not None:

            pos = torch.arange(T, device=idx.device)

            x = x + self.pos_emb(pos)[None, :, :]

        x = self.drop(x)


        for layer in self.layers:

            x = layer(x, self.freqs_cis)

        x = self.norm(x)


        if targets is None:

            # Inference: only the last position matters.

            logits = self.output(x[:, [-1], :])

            return logits, None


        logits = self.output(x)

        loss = F.cross_entropy(

            logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1

        )

        return logits, loss


    def configure_optimizers(self, weight_decay, lr, betas, device_type):

        """Decay matrices, do not decay norms and biases.


        Weight decay on a 1-D gain parameter just shrinks it toward zero for no

        benefit, so those are placed in a separate group.

        """

        decay, no_decay = [], []

        for p in self.parameters():

            if not p.requires_grad:

                continue

            (decay if p.dim() >= 2 else no_decay).append(p)

        groups = [

            {"params": decay, "weight_decay": weight_decay},

            {"params": no_decay, "weight_decay": 0.0},

        ]

        fused = device_type == "cuda"

        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)


    @torch.no_grad()

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=200, top_p=None):

        """Sample a continuation. temperature=0 gives greedy decoding."""

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.cfg.seq_len :]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]

            if temperature == 0.0:

                nxt = torch.argmax(logits, dim=-1, keepdim=True)

            else:

                logits = logits / temperature

                if top_p is not None:

                    # Nucleus: keep the smallest set whose cumulative

                    # probability reaches top_p, then renormalise.

                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)

                    probs_sorted = F.softmax(sorted_logits, dim=-1)

                    cum = torch.cumsum(probs_sorted, dim=-1)

                    # Drop everything after the token that crosses top_p,

                    # but always keep at least one token.

                    remove = cum - probs_sorted > top_p

                    remove[:, 0] = False

                    sorted_logits[remove] = -float("inf")

                    logits = torch.full_like(logits, -float("inf")).scatter(

                        1, sorted_idx, sorted_logits

                    )

                elif top_k is not None:

                    k = min(top_k, logits.size(-1))

                    v, _ = torch.topk(logits, k)

                    logits[logits < v[:, [-1]]] = -float("inf")

                probs = F.softmax(logits, dim=-1)

                nxt = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, nxt), dim=1)

        return idx
