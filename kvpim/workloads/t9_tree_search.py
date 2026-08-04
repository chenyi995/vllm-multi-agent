# SPDX-License-Identifier: Apache-2.0
"""T9 Tree Search on Game of 24, following Tree of Thoughts.

BFS of depth 3 with breadth b=5: one propose call per surviving state lists all
next steps at once, then one value call scores each candidate sure/likely/
impossible. Prompts are quoted verbatim from
`ref/tree-of-thought-llm/src/tot/prompts/game24.py`.

The paper samples the value three times at temperature 0.7; under this study's
`temperature=0` those three samples would be identical, so the value is taken
once (plan section 3).

Cache-wise this is the topology where siblings fan out from a shared path prefix
and a backtrack re-activates an older one.
"""

import random
import re
from collections.abc import Iterator

from kvpim.call import Call

# ref/tree-of-thought-llm/src/tot/prompts/game24.py::propose_prompt, verbatim.
PROPOSE_PROMPT = """Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: {input}
Possible next steps:
"""

# ref/tree-of-thought-llm/src/tot/prompts/game24.py::value_prompt, verbatim.
VALUE_PROMPT = """Evaluate if given numbers can reach 24 (sure/likely/impossible)
10 14
10 + 14 = 24
sure
11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
impossible
4 4 10
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
sure
4 9 11
9 + 11 + 4 = 20 + 4 = 24
sure
5 7 8
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
likely
5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
likely
10 10 11
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 10 are all too big
impossible
1 3 3
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
impossible
{input}
"""

VALUE_WEIGHTS = {"sure": 20.0, "likely": 1.0, "impossible": 0.001}
BREADTH = 5
DEPTH = 3
MAX_CANDIDATES = 8

_LEFT = re.compile(r"left:\s*([0-9.\s]+)\)")


def parse_candidates(reply: str, limit: int = MAX_CANDIDATES) -> list[tuple[str, str]]:
    """Extracts `(step line, remaining numbers)` pairs from a propose reply."""
    found = []
    for line in reply.strip().splitlines():
        match = _LEFT.search(line)
        if match:
            found.append((line.strip(), " ".join(match.group(1).split())))
        if len(found) >= limit:
            break
    return found


def score_value(reply: str) -> float:
    """Maps the value reply onto ToT's sure/likely/impossible weights."""
    tail = reply.strip().lower()
    for name in ("impossible", "likely", "sure"):
        if tail.endswith(name) or f"\n{name}" in tail:
            return VALUE_WEIGHTS[name]
    return VALUE_WEIGHTS["likely"] if "likely" in tail else VALUE_WEIGHTS["impossible"]


def build(
    task: dict, breadth: int = BREADTH, depth: int = DEPTH
) -> Iterator[Call]:
    """Yields the BFS propose/value stream for one Game of 24 puzzle."""
    # A state is (numbers still available, the steps taken to get there).
    frontier = [(task["puzzle"], "")]

    for level in range(depth):
        scored = []
        for node_idx, (numbers, path) in enumerate(frontier):
            proposal = yield Call(
                agent_id="expander",
                parent_idx=None,
                messages=[
                    {"role": "user", "content": PROPOSE_PROMPT.format(input=numbers)}
                ],
                meta={"stage": "propose", "level": level, "node": node_idx,
                      "numbers": numbers},
            )
            for step, remaining in parse_candidates(proposal):
                verdict = yield Call(
                    agent_id="evaluator",
                    parent_idx=None,
                    messages=[
                        {
                            "role": "user",
                            "content": VALUE_PROMPT.format(input=remaining),
                        }
                    ],
                    meta={"stage": "value", "level": level, "node": node_idx,
                          "numbers": remaining},
                )
                scored.append((score_value(verdict), remaining, path + step + "\n"))

        if not scored:
            break
        scored.sort(key=lambda item: item[0], reverse=True)
        frontier = [(numbers, path) for _, numbers, path in scored[:breadth]]


def load_tasks(num_tasks: int = 10, seed: int = 0) -> list[dict]:
    """Takes every tenth puzzle from ToT's 901-1000 evaluation slice.

    Even spacing covers both the early-stopping and the full-tree behaviour
    rather than clustering on the hardest tail (plan section 3).
    """
    import csv
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "ref/tree-of-thought-llm/src/tot/data/24/24.csv"
    )
    with path.open() as f:
        rows = list(csv.DictReader(f))

    del seed  # the slice is fixed by the paper; nothing is sampled at random
    ranks = list(range(905, 1000, 95 // max(num_tasks - 1, 1)))[:num_tasks]
    by_rank = {int(row["Rank"]): row for row in rows}
    return [
        {
            "task_id": f"game24_rank{rank}",
            "puzzle": by_rank[rank]["Puzzles"],
            "solved_rate": by_rank[rank]["Solved rate"],
        }
        for rank in ranks
        if rank in by_rank
    ]
