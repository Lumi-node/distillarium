import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLUGate(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.gelu(self.w_gate(x)) * self.w_up(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = q.shape
        S = k.shape[1]

        q = self.q_proj(q).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)


class AttentionOnlyLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.glu = GLUGate(d_model, d_model * 2)
        self.ln_self = nn.LayerNorm(d_model)
        self.ln_cross = nn.LayerNorm(d_model)
        self.ln_glu = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        schema: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.ln_self(x)
        x = x + self.self_attn(h, h, h, mask=causal_mask)
        h = self.ln_cross(x)
        x = x + self.cross_attn(h, schema, schema)
        h = self.ln_glu(x)
        x = x + self.glu(h)
        return x


class AttentionOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 384,
        n_heads: int = 8,
        n_layers: int = 8,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.schema_tok_emb = nn.Embedding(vocab_size, d_model)
        self.schema_pos_emb = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList(
            [AttentionOnlyLayer(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    def forward(
        self, input_ids: torch.Tensor, schema_ids: torch.Tensor
    ) -> torch.Tensor:
        B, L = input_ids.shape
        _, S = schema_ids.shape

        input_ids = input_ids.clamp(0, self.vocab_size - 1)
        schema_ids = schema_ids.clamp(0, self.vocab_size - 1)

        pos_input = torch.arange(L, device=input_ids.device).clamp(max=self.max_seq_len - 1).unsqueeze(0)
        pos_schema = torch.arange(S, device=schema_ids.device).clamp(max=self.max_seq_len - 1).unsqueeze(0)

        x = self.tok_emb(input_ids) + self.pos_emb(pos_input)
        schema = self.schema_tok_emb(schema_ids) + self.schema_pos_emb(pos_schema)

        causal_mask = self._causal_mask(L, input_ids.device)

        for layer in self.layers:
            x = layer(x, schema, causal_mask)

        x = self.ln_final(x)
        return self.lm_head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
