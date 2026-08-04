# SPDX-License-Identifier: Apache-2.0
"""Copy sampled-layer KV blocks out of the live cache (ample tier only).

`VLLM_ENABLE_V1_MULTIPROCESSING=0` puts the engine in the driver process, so the
per-layer tensors and the block pool are plain attributes we can read after each
call. A cached block is immutable until it is evicted, so every block is written
exactly once — the file named after the call that first materialised it.

FlashAttention packs both halves into one tensor. `flash_attn.py` derives the
kernel's views as `kv_cache.transpose(1, 2).split(head_size, dim=-1)`, i.e.
logical `(B, H, N, 2D)` becomes `(B, N, H, D)` for K and the same for V, K first.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import save_file

from vllm.v1.core.kv_cache_utils import get_block_hash, maybe_convert_block_hash


def _hexify(block_hash):
    return block_hash.hex() if isinstance(block_hash, bytes) else block_hash


def get_model_runner(llm):
    """Returns the in-process `GPUModelRunner`."""
    worker = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker
    return getattr(worker, "worker", worker).model_runner


def get_block_pool(llm):
    """Returns the scheduler's `BlockPool`."""
    manager = llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager
    coordinator = getattr(manager, "coordinator", None)
    pool = getattr(coordinator, "block_pool", None)
    return pool if pool is not None else manager.block_pool


def cached_blocks(pool) -> list[dict]:
    """Lists every block currently holding cached content.

    Returns:
        Dicts with ``block_id``, the event-stream ``block_hash`` and the
        exclusive end position ``num_tokens`` of the block within its sequence.
    """
    found = []
    for block in pool.blocks:
        key = block.block_hash
        if key is None or block.is_null:
            continue
        found.append(
            {
                "block_id": block.block_id,
                "block_hash": _hexify(maybe_convert_block_hash(get_block_hash(key))),
                "num_tokens": block.block_hash_num_tokens,
            }
        )
    return found


def split_kv(layer_tensor: torch.Tensor, head_size: int):
    """Splits a packed layer tensor into K and V views of ``(B, N, H, D)``."""
    return layer_tensor.transpose(1, 2).split(head_size, dim=-1)


@dataclass
class KVDumper:
    """Writes each newly cached block once, under the call that created it.

    Attributes:
        out_dir: ``dumps/`` root for the current configuration.
        sample_layers: Layer indices to persist.
        head_size: Head dimension, used to split the packed K/V tensor.
    """

    out_dir: Path
    sample_layers: tuple[int, ...]
    head_size: int = 128
    # Keyed by block hash, not block id: a block evicted and later recomputed
    # comes back on a different id, and dumping it twice would double-count it
    # in N_total even though it is one logical block. Within a workflow the hash
    # identifies content and prefix uniquely, which is exactly the identity
    # N_total is meant to count.
    seen: set = field(default_factory=set)
    num_blocks_written: int = 0

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def start_workflow(self) -> None:
        """Drops the seen-set at a workflow boundary.

        The cache is cleared between workflows, which both recycles block ids and
        makes a repeat of the same content a genuinely separate physical block —
        exactly the kind of duplicate pair this study counts. Both reasons say
        the set must not survive across workflows.
        """
        self.seen.clear()

    def dump_call(self, llm, workflow_id: str, call_idx: int) -> int:
        """Persists blocks cached since the previous call.

        Args:
            llm: The live `LLM` whose cache is being sampled.
            workflow_id: Directory name for this workflow.
            call_idx: Index of the call that has just completed.

        Returns:
            How many blocks this call contributed.
        """
        pool = get_block_pool(llm)
        runner = get_model_runner(llm)

        fresh = [b for b in cached_blocks(pool) if b["block_hash"] not in self.seen]
        if not fresh:
            return 0

        block_ids = torch.tensor([b["block_id"] for b in fresh], dtype=torch.long)
        workflow_dir = self.out_dir / workflow_id
        workflow_dir.mkdir(parents=True, exist_ok=True)

        for layer in self.sample_layers:
            key_cache, value_cache = split_kv(runner.kv_caches[layer], self.head_size)
            tensors = {
                "k": key_cache[block_ids].contiguous().cpu(),
                "v": value_cache[block_ids].contiguous().cpu(),
            }
            save_file(
                tensors,
                str(workflow_dir / f"{call_idx:04d}_{layer:02d}.safetensors"),
                metadata={"layer": str(layer), "call_idx": str(call_idx)},
            )

        # Positions are what make K_derope recoverable: `num_tokens` is the
        # exclusive end of the prefix this hash covers. With a `prefix_match_unit`
        # finer than the block, that end can land inside the block, so the start
        # is taken from the block boundary rather than by subtracting its size.
        # `positions[i]` is the token in slot `i` of the block.
        size = key_cache.shape[1]
        for block in fresh:
            end = block["num_tokens"]
            start = ((end - 1) // size) * size
            block["positions"] = list(range(start, end))
        (workflow_dir / f"{call_idx:04d}_meta.json").write_text(
            json.dumps(
                {
                    "workflow_id": workflow_id,
                    "call_idx": call_idx,
                    "layers": list(self.sample_layers),
                    "block_size": key_cache.shape[1],
                    "num_kv_heads": key_cache.shape[2],
                    "head_size": key_cache.shape[3],
                    "blocks": fresh,
                },
                indent=None,
            )
            + "\n"
        )

        self.seen.update(b["block_hash"] for b in fresh)
        self.num_blocks_written += len(fresh)
        return len(fresh)
