# SPDX-License-Identifier: Apache-2.0
"""Offline inverse of the NeoX-style rotary embedding vLLM applies to K.

The cache stores post-RoPE K, so `K_derope` is recovered analytically from the
dumped tensor plus the position each token sat at. Nothing here touches vLLM at
runtime; it mirrors `RotaryEmbedding._compute_cos_sin_cache` and
`ApplyRotaryEmb.forward_static` (neox branch) so the two stay bit-comparable.

Forward (neox, halves split at rotary_dim/2):
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
Inverse:
    x1 = o1 * cos + o2 * sin
    x2 = o2 * cos - o1 * sin
"""

from dataclasses import dataclass

import torch

QWEN3_ROPE_THETA = 5_000_000.0
QWEN3_HEAD_DIM = 128


@dataclass
class RopeParams:
    """Rotary parameters read off the model config."""

    head_size: int = QWEN3_HEAD_DIM
    rotary_dim: int = QWEN3_HEAD_DIM
    base: float = QWEN3_ROPE_THETA

    @classmethod
    def from_hf_config(cls, config: dict) -> "RopeParams":
        if config.get("rope_scaling"):
            raise NotImplementedError(
                "rope_scaling is not handled; Qwen3-4B-Instruct-2507 has none"
            )
        head_dim = config.get("head_dim") or (
            config["hidden_size"] // config["num_attention_heads"]
        )
        return cls(
            head_size=head_dim,
            rotary_dim=head_dim,
            base=float(config["rope_theta"]),
        )


def cos_sin(
    positions: torch.Tensor,
    params: RopeParams = RopeParams(),
    dtype: torch.dtype = torch.float32,
    cache_dtype: torch.dtype | None = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns cos/sin of shape ``[len(positions), rotary_dim // 2]``.

    Args:
        positions: Absolute token positions.
        params: Rotary parameters of the model.
        dtype: Arithmetic dtype for the returned tables.
        cache_dtype: vLLM builds `cos_sin_cache` in fp32 and then stores it in
            the model dtype (`RotaryEmbedding.__init__`). Rounding through the
            same dtype makes this the exact inverse of what was applied; pass
            None to keep full precision.
    """
    inv_freq = 1.0 / (
        params.base
        ** (
            torch.arange(0, params.rotary_dim, 2, dtype=torch.float)
            / params.rotary_dim
        )
    )
    freqs = torch.einsum("i,j->ij", positions.to(torch.float), inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    if cache_dtype is not None:
        cos, sin = cos.to(cache_dtype), sin.to(cache_dtype)
    return cos.to(dtype), sin.to(dtype)


def _rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool):
    """Applies (or undoes) the rotation on ``x[..., :rotary_dim]``."""
    rotary_dim = cos.shape[-1] * 2
    rotated, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    cos = cos.unsqueeze(-2).to(rotated.dtype)
    sin = sin.unsqueeze(-2).to(rotated.dtype)
    x1, x2 = torch.chunk(rotated, 2, dim=-1)
    if inverse:
        o1 = x1 * cos + x2 * sin
        o2 = x2 * cos - x1 * sin
    else:
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2, passthrough), dim=-1)


def derope(
    k_post: torch.Tensor,
    positions: torch.Tensor,
    params: RopeParams = RopeParams(),
    cache_dtype: torch.dtype | None = torch.bfloat16,
) -> torch.Tensor:
    """Removes the rotation from post-RoPE keys.

    Args:
        k_post: ``[seq_len, num_kv_heads, head_size]`` as stored in the cache.
        positions: ``[seq_len]`` absolute position of each token in its request.
        params: Rotary parameters of the model that produced ``k_post``.
        cache_dtype: Dtype vLLM rounded its cos/sin table through.

    Returns:
        Position-independent keys, same shape and dtype as ``k_post``.
    """
    work = k_post.float()
    cos, sin = cos_sin(positions, params, cache_dtype=cache_dtype)
    return _rotary(work, cos, sin, inverse=True).to(k_post.dtype)


def rope(
    k_pre: torch.Tensor,
    positions: torch.Tensor,
    params: RopeParams = RopeParams(),
    cache_dtype: torch.dtype | None = torch.bfloat16,
) -> torch.Tensor:
    """Forward rotation; used to close the loop in sanity check #1."""
    work = k_pre.float()
    cos, sin = cos_sin(positions, params, cache_dtype=cache_dtype)
    return _rotary(work, cos, sin, inverse=False).to(k_pre.dtype)
