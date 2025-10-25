import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    n_heads: int
    n_kv_heads: int
    n_kv_groups: int
    head_dim: int
    n_layers: int
    attention_bias: bool
    intermediate_size: int
    mlp_bias: bool
    eps: float
    dropout: float
    max_position_embeddings: int
    pre_norm: bool
    tie_weights: bool
    max_seq_len: int


class RMSNorm(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_position_embeddings=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        cos = self.cos[:, :, :seq_len, :].to(device=device, dtype=dtype)
        sin = self.sin[:, :, :seq_len, :].to(device=device, dtype=dtype)
        return cos, sin


def apply_rotary(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rot = torch.stack([-x2, x1], dim=-1).reshape_as(x)
    return (x * cos) + (x_rot * sin)


class GroupedMultiQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.attention_bias = config.attention_bias
        self.dropout = nn.Dropout(config.dropout)

        if self.n_heads * self.head_dim != self.hidden_size:
            raise ValueError("hidden_size must equal n_heads * head_dim")

        # derive n_kv_groups if None
        if config.n_kv_groups is None:
            if self.n_kv_heads == 0:
                raise ValueError("n_kv_heads must be > 0")
            self.n_kv_groups = self.n_heads // self.n_kv_heads
            if self.n_heads % self.n_kv_heads != 0:
                raise ValueError("n_heads must be divisible by n_kv_heads to derive groups")
        else:
            self.n_kv_groups = config.n_kv_groups
            if self.n_kv_heads * self.n_kv_groups != self.n_heads:
                raise ValueError("n_heads must equal n_kv_heads * n_kv_groups")

        self.q_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=self.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=self.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=self.attention_bias)
        self.w_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.max_position_embeddings)

    def forward(self, x):
        B, T, _ = x.shape
        device = x.device
        dtype = x.dtype

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(T, device=device, dtype=dtype)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if self.n_kv_groups != 1:
            k = k.repeat_interleave(self.n_kv_groups, dim=1)
            v = v.repeat_interleave(self.n_kv_groups, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # causal mask
        mask = torch.triu(torch.full((T, T), float("-inf"), device=device, dtype=dtype), diagonal=1)
        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        return self.w_o(out)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.dropout = nn.Dropout(config.dropout)

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(self.dropout(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = GroupedMultiQueryAttention(config)
        self.feed_forward = SwiGLUFeedForward(config)
        self.attn_norm = RMSNorm(config)
        self.ffn_norm = RMSNorm(config)
        self.dropout = nn.Dropout(config.dropout)
        self.pre_norm = config.pre_norm

    def forward(self, x):
        if self.pre_norm:
            x = x + self.dropout(self.attention(self.attn_norm(x)))
            x = x + self.dropout(self.feed_forward(self.ffn_norm(x)))
        else:
            x = self.attn_norm(x + self.dropout(self.attention(x)))
            x = self.ffn_norm(x + self.dropout(self.feed_forward(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.final_norm = RMSNorm(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(max(1, self.config.n_layers)))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        x = self.token_embedding(input_ids) * math.sqrt(self.config.hidden_size)
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0, filter_value: float = -float('Inf')) -> torch.Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    This is taken from common implementations (Hugging Face transformers style).
    Args:
        logits: logits distribution shape (batch, vocab)
        top_k: keep only top k tokens with highest probability (0 = no top-k)
        top_p: keep the top tokens with cumulative probability >= top_p (0.0 = no nucleus)
        filter_value: value to set for filtered logits
    Returns:
        filtered logits with the same shape
    """
    top_k = max(top_k, 0)
    batch_size, vocab_size = logits.size()

    if top_k > 0:
        # Remove all tokens with a probability less than the top-k tokens
        top_k = min(max(top_k, 1), vocab_size)
        values_to_keep, _ = torch.topk(logits, top_k)
        min_values = values_to_keep[:, -1].unsqueeze(1).expand_as(logits)
        logits = torch.where(logits < min_values, torch.full_like(logits, filter_value), logits)

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_mask = cumulative_probs > top_p

        # Shift the mask right to keep at least one token
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


@torch.no_grad()
def generate(
    model: Transformer,
    input_ids: torch.LongTensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    do_sample: bool = True,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    device: Optional[torch.device] = None,
):
    """
    Autoregressive generation helper for the model. This implementation does NOT use KV cache
    (the model defined in this file does not implement a cache), so generation is performed
    by repeatedly calling the model on the growing sequence. It supports temperature,
    top-k and nucleus (top-p) sampling, greedy decoding, and optional early stopping
    on an `eos_token_id`.

    Args:
        model: the Transformer instance
        input_ids: (batch, seq_len) input token ids
        max_new_tokens: number of tokens to generate
        temperature: sampling temperature (<=0 or do_sample=False => greedy)
        top_k: top-k filtering (0 disables)
        top_p: nucleus/top-p filtering (0.0 disables)
        do_sample: whether to sample (True) or do greedy decoding (False)
        eos_token_id: optional EOS id to stop generation for individual sequences
        pad_token_id: optional pad id to use for finished sequences
        device: optional torch.device to run on; if None uses model's device

    Returns:
        tensor of shape (batch, seq_len + generated) with generated tokens appended
    """
    model.eval()
    if device is None:
        # try to infer device
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

    input_ids = input_ids.to(device)

    batch_size, seq_len = input_ids.shape
    generated = 0
    unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        logits = model(input_ids)
        # logits shape: (batch, seq_len_total, vocab)
        next_token_logits = logits[:, -1, :]

        if temperature <= 0 or not do_sample:
            # Greedy
            next_tokens = torch.argmax(next_token_logits, dim=-1)
        else:
            logits_proc = next_token_logits / max(temperature, 1e-8)
            logits_proc = top_k_top_p_filtering(logits_proc, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits_proc, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # If EOS is provided, update finished sequences and pad further tokens
        if eos_token_id is not None:
            is_eos = next_tokens.eq(eos_token_id)
            # sequences that have just finished
            just_finished = unfinished & is_eos
            unfinished = unfinished & (~is_eos)

        # For sequences already finished, append pad_token_id (if provided), otherwise keep EOS or sampled token
        if pad_token_id is not None and not unfinished.all():
            finished_mask = ~unfinished
            if finished_mask.any():
                next_tokens = next_tokens.masked_fill(finished_mask, pad_token_id)

        # append
        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=1)
        generated += 1

        if eos_token_id is not None and not unfinished.any():
            break

    return input_ids


def _smoke_test():
    config = ModelConfig(
        vocab_size=128,
        hidden_size=64,
        n_heads=4,
        n_kv_heads=4,
        n_kv_groups=None,
        head_dim=16,
        n_layers=2,
        attention_bias=False,
        intermediate_size=256,
        mlp_bias=False,
        eps=1e-5,
    )
    model = Transformer(config)
    model.eval()

    batch, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    logits, loss = model(input_ids, targets=input_ids)

    assert logits.shape == (batch, seq_len, config.vocab_size)
    assert loss.dim() == 0
    print("Smoke test passed: logits shape", logits.shape, "loss", loss.detach().item())


if __name__ == "__main__":
    _smoke_test()

