import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from flash_attn.modules.mha import FlashSelfAttention, FlashCrossAttention
from flash_attn.layers.rotary import apply_rotary_emb, apply_rotary_emb_qkv_, apply_rotary_emb_kv_
from einops import rearrange


class RotaryEmbedding(nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(
        self,
        dim: int,
        base=10000.0,
        pos_idx_in_fp32=True,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_inv_freq(self, device=None):
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )
    
    def get_cos_sin(self, q_t, k_t=None, device=None, dtype=None):
        if device is None:
            device = q_t.device
        if self.pos_idx_in_fp32:
            q_t = q_t.to(device=device, dtype=torch.float32)
            if k_t is not None:
                k_t = k_t.to(device=device, dtype=torch.float32)
            if self.inv_freq.dtype != torch.float32:
                inv_freq = self._compute_inv_freq(device=device)
            else:
                inv_freq = self.inv_freq
        else:
            q_t = q_t.to(device=device, dtype=dtype)
            if k_t is not None:
                k_t = k_t.to(device=device, dtype=torch.dtype)
            inv_freq = self.inv_freq
        q_freqs = torch.outer(q_t, inv_freq)
        if k_t is not None:
            k_freqs = torch.outer(k_t, inv_freq)
        _cos = torch.cos(q_freqs).to(dtype)
        _sin = torch.sin(q_freqs).to(dtype)
        _cos_k = torch.cos(k_freqs).to(dtype) if k_t is not None else None
        _sin_k = torch.sin(k_freqs).to(dtype) if k_t is not None else None
        return _cos, _sin, _cos_k, _sin_k

    def forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        q_t: Optional[torch.Tensor] = None,
        k_t: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        qkv: (batch, seqlen, 3, nheads, headdim) or (batch, seqlen, num_heads_q + 2 * num_heads_k, headdim)
            if kv is none, else it's just q of shape (batch, seqlen, nheads, headdim).
            If qkv has shape (batch, seqlen, num_heads_q + 2 * num_heads_k, headdim) (e.g. MQA / GQA),
            then num_heads_q must be provided.
        kv: (batch, seqlen, 2, nheads, headdim)
        Apply rotary embedding *inplace* to qkv and / or kv.
        """
        _cos, _sin, _cos_k, _sin_k = self.get_cos_sin(q_t, k_t, device=qkv.device, dtype=qkv.dtype)
        if kv is None:  # self-attn
            qkv = apply_rotary_emb_qkv_(
                qkv,
                _cos,
                _sin,
            )
            del _cos, _sin, _cos_k, _sin_k
            torch.cuda.empty_cache()
            return qkv
        else:  # cross-attn
            q = qkv
            q = apply_rotary_emb(
                q,
                _cos,
                _sin,
                inplace=True,
            )
            kv = apply_rotary_emb_kv_(
                kv,
                _cos_k,
                _sin_k,
            )
            del _cos, _sin, _cos_k, _sin_k
            torch.cuda.empty_cache()
            return q, kv


class simpleMHA(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_kv=None,
        causal=False,
        cross_attn=False,
        rope_theta=10000.0,
        checkpointing=False,
        device=None,
    ) -> None:
        """
        num_heads_kv: can be used to toggle MQA / GQA. If None, use num_heads.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attn = cross_attn
        self.causal = causal
        self.rotary_emb_dim = embed_dim // num_heads // 2
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        self.head_dim = embed_dim // num_heads
        self.checkpointing = checkpointing
        
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads

        self.rotary_emb = RotaryEmbedding(
            self.rotary_emb_dim,
            base=rope_theta,
            device=device,
        )
        
        if cross_attn: # cross-attention
            self.attn = FlashCrossAttention(causal=False)
            self.Wq = nn.Linear(embed_dim, self.head_dim * self.num_heads)
            self.Wkv = nn.Linear(embed_dim, self.head_dim * self.num_heads_kv * 2)
        else: # self-attention
            self.attn = FlashCrossAttention(causal=causal)
            self.Wqkv = nn.Linear(embed_dim, self.head_dim * self.num_heads + self.head_dim * self.num_heads_kv * 2)
        self.Wo = nn.Linear(embed_dim, embed_dim)
        
    def forward(
        self,
        x_q,
        x_kv=None,
        q_position_ids=None,
        k_position_ids=None,
    ):
        if self.cross_attn:
            if x_kv is None:
                x_kv = x_q
            if q_position_ids is None:
                q_position_ids = torch.arange(x_q.shape[1], device=x_q.device)
            if k_position_ids is None:
                k_position_ids = torch.arange(x_kv.shape[1], device=x_kv.device)
            
            q = self.Wq(x_q)
            kv = self.Wkv(x_kv)
            q = rearrange(q.unsqueeze(0), "... (h d) -> ... h d", d=self.head_dim)
            kv = rearrange(kv.unsqueeze(0), "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)
            q, kv = self.rotary_emb(q, kv=kv, q_t=q_position_ids, k_t=k_position_ids)
            if self.checkpointing:
                q = torch.utils.checkpoint.checkpoint(lambda q, kv: self.attn(q=q, kv=kv), q, kv)
            else:
                q = self.attn(q, kv)
            q = q.flatten(-2)
        else:
            qkv = self.Wqkv(x_q)
            q = qkv[..., : self.num_heads * self.head_dim]
            kv = qkv[..., self.num_heads * self.head_dim :]
            if q_position_ids is None:
                q_position_ids = torch.arange(x_q.shape[1], device=x_q.device)
            q = rearrange(q.unsqueeze(0), "... (h d) -> ... h d", d=self.head_dim)
            kv = rearrange(kv.unsqueeze(0), "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)
            
            q, kv = self.rotary_emb(q, kv=kv, q_t=q_position_ids, k_t=q_position_ids)
            if self.checkpointing:
                q = torch.utils.checkpoint.checkpoint(lambda q, kv: self.attn(q=q, kv=kv), q, kv)
            else:
                q = self.attn(q, kv)
            q = q.flatten(-2)
        q = self.Wo(q)
        return q.squeeze(0)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        self.self_attn = simpleMHA(config.hidden_size, config.num_attention_heads, num_heads_kv=config.num_key_value_heads, rope_theta=config.rope_theta, cross_attn=False, causal=False, checkpointing=True)
        self.cross_attn = simpleMHA(config.hidden_size, config.num_attention_heads, num_heads_kv=config.num_key_value_heads, rope_theta=config.rope_theta, cross_attn=True, checkpointing=True)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )
        
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        self.norm3 = nn.LayerNorm(self.hidden_size)
    
    def forward(self, x_q, x_kv, q_position_ids=None, k_position_ids=None, *args, **kwargs):
        # self-attention
        residual = x_q
        x_q = self.self_attn(x_q, q_position_ids=q_position_ids)
        x_q = self.norm1(residual + x_q)
        
        # cross-attention
        residual = x_q
        x_q = self.cross_attn(x_q=x_q, x_kv=x_kv, q_position_ids=q_position_ids, k_position_ids=k_position_ids)
        x_q = self.norm2(residual + x_q)
        
        # feed-forward network
        residual = x_q
        x_q = self.mlp(x_q)
        x_q = self.norm3(residual + x_q)
        
        return x_q


class MemAug(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.layers = nn.ModuleList([Block(config) for _ in range(config.memaug_depth)])
        self.config = config
    
    def forward(self, x_q, x_kv=None, q_position_ids=None, k_position_ids=None, *args, **kwargs):
        for layer in self.layers:
            x_q = layer(x_q, x_kv, q_position_ids=q_position_ids, k_position_ids=k_position_ids)
        return x_q