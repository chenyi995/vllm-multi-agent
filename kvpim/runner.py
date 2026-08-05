# SPDX-License-Identifier: Apache-2.0
"""Runs one experiment configuration: topology x block tier x capacity tier.

The driver tokenizes every call itself and submits ``prompt_token_ids``, so the
token positions recorded alongside a KV dump are exactly the ones the model saw.
"""

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from kvpim.call import Call
from kvpim.events import KVEventCollector

SAMPLE_LAYERS = (0, 8, 17, 26, 35)

# Physical 4/8-token blocks are rejected by every CUDA attention backend, so the
# two finest tiers run a 16-token physical block with a finer matching unit.
_TIER_TO_PHYSICAL = {4: 16, 8: 16, 16: 16, 32: 32, 64: 64}

WorkloadFn = Callable[[dict], Iterator[Call]]


@dataclass
class RunConfig:
    """One cell of the 9 x 5 x 2 matrix."""

    topology: str
    block_tier: int
    capacity_tier: str
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    traces_root: Path = Path("/home/cw636/chenyi/KVPIM/traces")
    num_gpu_blocks_override: int | None = None
    # 0.95 rather than the usual 0.90: it buys ~9% more pool (93k -> ~102k
    # tokens on a 24GB A5000), which is what lets both capacity tiers run on the
    # same card and keeps the Q3 join from crossing a GPU architecture.
    gpu_memory_utilization: float = 0.95
    # The model ships a 262144 context; vLLM refuses to start unless the pool
    # can hold one full-length request. Capped well above any single call in
    # these workloads and below the A5000 pool. With an override in play the
    # admission check runs against the *overridden* pool, so the limited tier
    # must lower this too.
    max_model_len: int = 65536
    # MoA's own setting is 2048, which almost never binds (measured outputs
    # 291-1365) and leaves a workflow needing up to 135k tokens — more than a
    # 24GB card holds. 1024 truncates only the tail, keeping W around 92k.
    # Recorded confound: shorter references raise the share of the prompt taken
    # by the fixed-length persona and aggregate prompt, which can inflate the
    # duplicate fraction.
    max_tokens: int = 1024
    seed: int = 0
    zmq_port: int = 5557
    sample_layers: tuple[int, ...] = SAMPLE_LAYERS
    # Blocks are evicted at unpredictable moments once capacity bites, so the
    # limited tier records events only (plan section 6.3).
    dump: bool | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.block_tier not in _TIER_TO_PHYSICAL:
            raise ValueError(f"block_tier must be one of {list(_TIER_TO_PHYSICAL)}")
        if self.capacity_tier not in ("ample", "limited"):
            raise ValueError("capacity_tier must be 'ample' or 'limited'")
        if self.capacity_tier == "limited" and self.num_gpu_blocks_override is None:
            raise ValueError(
                "limited tier needs num_gpu_blocks_override = 0.5 * W / block_size"
            )
        self.traces_root = Path(self.traces_root)

    @property
    def dump_enabled(self) -> bool:
        """Ample tier only, and only where the tensors are not a duplicate set.

        Tiers 4/8/16 all run 16-token physical blocks and differ solely in
        matching granularity, so the 4 and 8 tiers reuse the 16 tier's tensors
        and are sliced into finer units offline (plan section 6.1).
        """
        if self.dump is not None:
            return self.dump
        return self.capacity_tier == "ample" and self.block_tier not in (4, 8)

    @property
    def block_size(self) -> int:
        return _TIER_TO_PHYSICAL[self.block_tier]

    @property
    def prefix_match_unit(self) -> int | None:
        return self.block_tier if self.block_tier in (4, 8) else None

    @property
    def out_dir(self) -> Path:
        return self.traces_root / f"{self.topology}_b{self.block_tier}_{self.capacity_tier}"


def _git_commit() -> str:
    repo = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tokenize(tokenizer, messages: list[dict], raw_completion: bool = False) -> list[int]:
    """Turns a call's messages into the token ids the engine will receive.

    transformers 5.x returns a `BatchEncoding` from `apply_chat_template`;
    older versions returned the id list directly.

    Args:
        tokenizer: The run's tokenizer.
        messages: The call's chat messages.
        raw_completion: Send the last message's text as a bare continuation
            instead of wrapping it in the chat template. Few-shot prompts that
            were written for completion models (Tree of Thoughts) are answered
            conversationally by an instruct model once the template frames them
            as a question, which destroys the format the parser depends on.
    """
    if raw_completion:
        text = messages[-1]["content"]
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    return list(ids)


def _template_hash(tokenizer) -> str:
    template = getattr(tokenizer, "chat_template", None) or ""
    return hashlib.sha256(template.encode()).hexdigest()[:16]


def build_llm(cfg: RunConfig):
    """Builds an in-process engine with KV events on.

    ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` keeps engine and driver in one process,
    which is what makes the KV tensors reachable for dumping.
    """
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # The installed flashinfer build targets CUDA 13 and would JIT-compile at
    # first sample; greedy decoding does not need it.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    from vllm import LLM
    from vllm.config import KVEventsConfig

    kwargs = dict(
        model=cfg.model,
        enforce_eager=True,
        dtype="auto",
        kv_cache_dtype="auto",
        seed=cfg.seed,
        max_model_len=cfg.max_model_len,
        # Qwen3's generation_config.json carries temperature 0.7 / top_k 20 /
        # top_p 0.8; loading it would silently contradict the temperature=0
        # protocol, so take vLLM's neutral defaults instead.
        generation_config="vllm",
        block_size=cfg.block_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        enable_prefix_caching=True,
        kv_events_config=KVEventsConfig(
            enable_kv_cache_events=True,
            publisher="zmq",
            endpoint=f"tcp://*:{cfg.zmq_port}",
            replay_endpoint=f"tcp://*:{cfg.zmq_port + 1}",
            topic="kv-events",
            buffer_steps=1_000_000,
            hwm=1_000_000,
            max_queue_size=1_000_000,
        ),
    )
    if cfg.prefix_match_unit is not None:
        kwargs["prefix_match_unit"] = cfg.prefix_match_unit
    if cfg.num_gpu_blocks_override is not None:
        kwargs["num_gpu_blocks_override"] = cfg.num_gpu_blocks_override
    return LLM(**kwargs)


def working_set_tokens(blocks_jsonl: str | Path) -> dict:
    """Working set W, per workflow.

    The cache is cleared between workflows, so W is a per-workflow quantity and
    `AllBlocksCleared` marks the boundaries. The limited tier sizes its pool from
    `max`, the workflow that must still fit.

    Returns:
        Per-workflow token counts plus their max, mean and number of evictions.
    """
    segments: list[dict[object, int]] = []
    current: dict[object, int] = {}
    num_removed = 0
    with Path(blocks_jsonl).open() as f:
        for line in f:
            record = json.loads(line)
            if record["type"] == "AllBlocksCleared":
                if current:
                    segments.append(current)
                current = {}
            elif record["type"] == "BlockRemoved":
                num_removed += len(record["block_hashes"])
            elif record["type"] == "BlockStored":
                for block_hash in record["block_hashes"]:
                    current.setdefault(block_hash, record["block_size"])
    if current:
        segments.append(current)

    per_workflow = [sum(seg.values()) for seg in segments]
    return {
        "per_workflow": per_workflow,
        "max": max(per_workflow) if per_workflow else 0,
        "mean": sum(per_workflow) // len(per_workflow) if per_workflow else 0,
        "num_blocks_removed": num_removed,
    }


def run(
    cfg: RunConfig,
    workload: WorkloadFn,
    tasks: list[dict],
    llm=None,
) -> Path:
    """Runs every task through one topology driver and writes the trace.

    Args:
        cfg: The configuration cell to run.
        workload: Generator factory; yields `Call`s and receives reply text back.
        tasks: The 10 sampled benchmark items for this topology.
        llm: Pre-built engine to reuse; built from ``cfg`` when omitted.

    Returns:
        The output directory holding manifest/calls/blocks for this run.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = out_dir / "blocks.jsonl"
    calls_path = out_dir / "calls.jsonl"
    for stale in (blocks_path, calls_path):
        stale.unlink(missing_ok=True)

    owns_llm = llm is None
    if owns_llm:
        llm = build_llm(cfg)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=0.0, top_p=1.0, top_k=0, max_tokens=cfg.max_tokens
    )

    collector = KVEventCollector(
        blocks_path,
        endpoint=f"tcp://localhost:{cfg.zmq_port}",
        replay_endpoint=f"tcp://localhost:{cfg.zmq_port + 1}",
    )
    collector.start()
    llm.reset_prefix_cache()

    dumper = None
    if cfg.dump_enabled:
        from kvpim.dump import KVDumper

        dumper = KVDumper(out_dir / "dumps", cfg.sample_layers)

    started = time.time()
    num_calls = 0
    with calls_path.open("w", buffering=1) as calls_file:
        for task_idx, task in enumerate(tasks):
            workflow_id = f"{cfg.topology}_w{task_idx:02d}"
            # One workflow's working set fits the pool, ten do not, so the ample
            # tier is defined per workflow: clear between them and require
            # `BlockRemoved` to stay empty within each (plan section 6.2).
            llm.reset_prefix_cache()
            if dumper is not None:
                dumper.start_workflow()
            generator = workload(task)
            reply = None
            call_idx = 0
            while True:
                try:
                    call = generator.send(reply) if call_idx else next(generator)
                except StopIteration:
                    break
                call.topology = cfg.topology
                call.workflow_id = workflow_id
                call.call_idx = call_idx

                prompt_token_ids = _tokenize(
                    tokenizer, call.messages, call.meta.get("raw_completion", False)
                )
                call_sampling = sampling
                if call.meta.get("stop") or call.meta.get("max_tokens"):
                    call_sampling = SamplingParams(
                        temperature=0.0,
                        top_p=1.0,
                        top_k=0,
                        max_tokens=call.meta.get("max_tokens", cfg.max_tokens),
                        stop=call.meta.get("stop"),
                    )
                # Bracketing timestamps let the offline reconciliation replay the
                # event stream up to the moment this call was admitted.
                t_start = time.time()
                output = llm.generate(
                    TokensPrompt(prompt_token_ids=prompt_token_ids),
                    call_sampling,
                    use_tqdm=False,
                )[0]
                t_end = time.time()
                completion = output.outputs[0]
                reply = completion.text
                if dumper is not None:
                    dumper.dump_call(llm, workflow_id, call_idx)

                calls_file.write(
                    json.dumps(
                        {
                            "topology": call.topology,
                            "workflow_id": call.workflow_id,
                            "call_idx": call.call_idx,
                            "agent_id": call.agent_id,
                            "parent_idx": call.parent_idx,
                            "prompt_token_ids": prompt_token_ids,
                            "output_token_ids": list(completion.token_ids),
                            "num_cached_tokens": output.num_cached_tokens,
                            "finish_reason": completion.finish_reason,
                            "t_start": t_start,
                            "t_end": t_end,
                            "meta": call.meta,
                        }
                    )
                    + "\n"
                )
                call_idx += 1
                num_calls += 1

    collector.stop()
    elapsed = time.time() - started

    manifest = {
        "topology": cfg.topology,
        "block_tier": cfg.block_tier,
        "block_size": cfg.block_size,
        "prefix_match_unit": cfg.prefix_match_unit,
        "capacity_tier": cfg.capacity_tier,
        "num_gpu_blocks_override": cfg.num_gpu_blocks_override,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "max_model_len": cfg.max_model_len,
        "working_set": working_set_tokens(blocks_path),
        "vllm_commit": _git_commit(),
        "model": cfg.model,
        "model_revision": getattr(tokenizer, "name_or_path", cfg.model),
        "chat_template_sha256_16": _template_hash(tokenizer),
        "tokenizer_class": type(tokenizer).__name__,
        "seed": cfg.seed,
        "max_tokens": cfg.max_tokens,
        "sample_layers": list(cfg.sample_layers),
        "num_workflows": len(tasks),
        "num_calls": num_calls,
        "num_events": collector.num_events,
        "num_missed_events": collector.num_missed,
        "dump_enabled": cfg.dump_enabled,
        "num_blocks_dumped": dumper.num_blocks_written if dumper else 0,
        "reset_between_workflows": True,
        "elapsed_s": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "extra": cfg.extra,
    }
    manifest["ample_criterion_met"] = (
        cfg.capacity_tier != "ample"
        or manifest["working_set"]["num_blocks_removed"] == 0
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if owns_llm:
        del llm
    return out_dir
