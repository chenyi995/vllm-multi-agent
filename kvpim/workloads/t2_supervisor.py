# SPDX-License-Identifier: Apache-2.0
"""T2 Supervisor / Orchestrator-Worker on GAIA.

Structure per plan section 3: the supervisor decomposes the question, one worker
runs per subtask, then the supervisor synthesises. The orchestrator's context
grows monotonically while the workers share the task-description prefix and fork
only at their own subtask text.

GAIA does not prescribe an agent structure, and with no tool environment the
workers answer from parametric knowledge — answer accuracy does not enter the KV
statistics (plan section 3).
"""

import random
import re
from collections.abc import Iterator

from kvpim.call import Call

SUPERVISOR_SYSTEM = (
    "You are the supervisor of a research team. Break the user's question into "
    "independent subtasks that specialists can answer separately, then combine "
    "their findings into one final answer."
)
WORKER_SYSTEM = (
    "You are a specialist worker. Answer only the subtask you are given, using "
    "the shared task description for context. Be factual and concise."
)

MAX_SUBTASKS = 5
_NUMBERED = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.{8,})$", re.MULTILINE)


def parse_subtasks(plan_text: str, limit: int = MAX_SUBTASKS) -> list[str]:
    """Pulls the numbered or bulleted subtasks out of the supervisor's plan.

    Falls back to one subtask holding the whole plan when nothing parses, so a
    malformed plan shortens the trace instead of dropping the workflow.
    """
    items = [m.group(1).strip() for m in _NUMBERED.finditer(plan_text)]
    if not items:
        items = [plan_text.strip()[:400] or "Answer the question directly."]
    return items[:limit]


def build(task: dict) -> Iterator[Call]:
    """Yields the supervisor/worker call stream for one GAIA question."""
    question = task["question"]
    shared_context = f"Task description:\n{question}"

    plan = yield Call(
        agent_id="supervisor",
        parent_idx=None,
        messages=[
            {"role": "system", "content": SUPERVISOR_SYSTEM},
            {"role": "user", "content": question},
        ],
        meta={"phase": "plan"},
    )

    subtasks = parse_subtasks(plan)
    results = []
    worker_indices = []
    for i, subtask in enumerate(subtasks):
        worker_indices.append(1 + i)
        reply = yield Call(
            agent_id=f"worker_{i}",
            parent_idx=0,
            messages=[
                {"role": "system", "content": WORKER_SYSTEM},
                {
                    "role": "user",
                    "content": f"{shared_context}\n\nYour subtask: {subtask}",
                },
            ],
            meta={"phase": "work", "worker": i, "subtask": subtask[:120]},
        )
        results.append(f"Worker {i} — {subtask}\n{reply}")

    yield Call(
        agent_id="supervisor",
        parent_idx=0,
        messages=[
            {"role": "system", "content": SUPERVISOR_SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": plan},
            {
                "role": "user",
                "content": "Findings from the workers:\n\n"
                + "\n\n".join(results)
                + "\n\nGive the final answer.",
            },
        ],
        meta={"phase": "synthesis", "parents": worker_indices},
    )


def load_tasks(num_tasks: int = 10, seed: int = 0) -> list[dict]:
    """Samples GAIA validation questions, 4/4/2 across levels 1/2/3.

    Attachment-bearing items are skipped: there is no tool environment, so the
    file would be invisible to the model and only the prompt matters here.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "gaia-benchmark/GAIA", "2023/validation/metadata.parquet", repo_type="dataset"
    )
    frame = pd.read_parquet(path)
    frame = frame[frame["file_name"].isna() | (frame["file_name"] == "")]

    quota = {"1": 4, "2": 4, "3": 2}
    if num_tasks != 10:
        share = max(num_tasks // 3, 1)
        quota = {"1": share, "2": share, "3": num_tasks - 2 * share}

    rng = random.Random(seed)
    picked = []
    for level, count in quota.items():
        pool = frame[frame["Level"] == level].sort_values("task_id")
        rows = list(pool.itertuples())
        picked.extend(rng.sample(rows, min(count, len(rows))))

    return [
        {
            "task_id": row.task_id,
            "question": row.Question,
            "level": row.Level,
            "final_answer": getattr(row, "_4", None),
        }
        for row in picked
    ]
