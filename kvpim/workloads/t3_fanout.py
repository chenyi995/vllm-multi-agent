# SPDX-License-Identifier: Apache-2.0
"""T3 Fan-out / Mixture-of-Agents on AlpacaEval 2.0.

Follows the MoA paper's layered structure (3 layers x 6 proposers, then one
aggregator) and its Aggregate-and-Synthesize injection verbatim: references are
appended to the system message, never to the user turn.

Single-model adaptation (plan section 3): one Qwen3-4B plays all six proposers
through six distinct system prompts, keeping "an agent is a system prompt".
"""

import random
from collections.abc import Iterator

from kvpim.call import Call

# ref/MoA/utils.py::inject_references_to_messages, quoted exactly.
AGGREGATE_AND_SYNTHESIZE = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""  # noqa: E501

PROPOSER_SYSTEM_PROMPTS = [
    "You are a concise expert. Answer directly and keep it tight.",
    "You are a rigorous analyst. Reason carefully and justify your claims.",
    "You are a practical assistant. Give actionable, step-by-step guidance.",
    "You are a creative thinker. Offer an original angle on the request.",
    "You are a meticulous editor. Prioritise precision, clarity and structure.",
    "You are a broad generalist. Cover the important aspects comprehensively.",
]

NUM_PROPOSERS = len(PROPOSER_SYSTEM_PROMPTS)
NUM_LAYERS = 3


def _with_references(system_prompt: str | None, references: list[str]) -> str:
    injected = AGGREGATE_AND_SYNTHESIZE
    for i, reference in enumerate(references):
        injected += f"\n{i + 1}. {reference}"
    if system_prompt is None:
        return injected
    return system_prompt + "\n\n" + injected


def build(task: dict, num_layers: int = NUM_LAYERS) -> Iterator[Call]:
    """Yields the MoA call stream for one AlpacaEval instruction.

    Args:
        task: An item with an ``instruction`` field.
        num_layers: Proposer layers before the aggregator; 2 gives MoA-Lite.

    Yields:
        Proposer calls layer by layer, then the aggregator call.
    """
    instruction = task["instruction"]
    user_turn = {"role": "user", "content": instruction}

    references: list[str] = []
    # A layer consumes every reply of the previous one, so the fan-in is kept in
    # `meta["parents"]`; `parent_idx` carries the first of them.
    parents: list[int] = []
    next_idx = 0

    for layer in range(num_layers):
        replies = []
        layer_indices = []
        for i, proposer_system in enumerate(PROPOSER_SYSTEM_PROMPTS):
            system = (
                proposer_system
                if layer == 0
                else _with_references(proposer_system, references)
            )
            layer_indices.append(next_idx)
            next_idx += 1
            reply = yield Call(
                agent_id=f"proposer_{i}",
                parent_idx=parents[0] if parents else None,
                messages=[{"role": "system", "content": system}, user_turn],
                meta={"layer": layer, "proposer": i, "parents": list(parents)},
            )
            replies.append(reply)
        references = replies
        parents = layer_indices

    yield Call(
        agent_id="aggregator",
        parent_idx=parents[0] if parents else None,
        messages=[
            {"role": "system", "content": _with_references(None, references)},
            user_turn,
        ],
        meta={"layer": num_layers, "role": "aggregator", "parents": list(parents)},
    )


def load_tasks(num_tasks: int = 10, seed: int = 0) -> list[dict]:
    """Samples instructions from AlpacaEval 2.0, stratified by length.

    Args:
        num_tasks: How many instructions to draw.
        seed: Fixed seed so the same items are replayed across configurations.

    Returns:
        Items carrying ``instruction`` plus their index in the source split.
    """
    import json

    from huggingface_hub import hf_hub_download

    # The dataset ships a loading script, which `datasets` 5.x refuses to run;
    # the 805-instruction eval set is the raw JSON next to it.
    path = hf_hub_download(
        "tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset"
    )
    split = json.load(open(path))
    order = sorted(range(len(split)), key=lambda i: len(split[i]["instruction"]))
    rng = random.Random(seed)
    bucket_size = len(order) / num_tasks
    picks = [
        order[rng.randrange(int(b * bucket_size), int((b + 1) * bucket_size))]
        for b in range(num_tasks)
    ]
    return [
        {
            "task_id": f"alpaca_eval_{idx}",
            "instruction": split[idx]["instruction"],
            "source_index": idx,
            "source_dataset": split[idx]["dataset"],
        }
        for idx in sorted(picks)
    ]
