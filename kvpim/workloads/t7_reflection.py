# SPDX-License-Identifier: Apache-2.0
"""T7 Reflection / Generator-Critic on HumanEval, following Reflexion.

Three components in a loop — Actor, Evaluator, Self-Reflection — with the actor
prompted from the signature, its previous implementation, the test result and
the latest reflection. Memory keeps only the most recent reflection (Omega = 1)
and the official repo's default `max_iters=2` bounds the loop (plan section 3).

Generator and critic are siblings on the radix tree: both extend the same
history but under different system prompts.

The Evaluator runs the tests it writes, as in the paper: the verdict that drives
the loop comes from a real interpreter, not from a model's opinion. Execution is
confined to a subprocess in a scratch directory under a wall-clock, CPU and
address-space limit.

Evaluation problems come from a baseline pass over all 164 HumanEval problems
(`scratch/humaneval_baseline.py`), keeping those the actor fails on its first
attempt — a problem solved immediately produces no iteration trace. If that file
is absent the sampler falls back to canonical solution length as a proxy.
"""

import json
import random
import re
import resource
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

from kvpim.call import Call

ACTOR_SYSTEM = (
    "You are a Python programmer. Write the complete implementation for the "
    "given signature. Output only a single ```python code block."
)
EVALUATOR_SYSTEM = (
    "You are an evaluator. Write unit tests for the implementation as a series "
    "of bare `assert` statements that call the function directly. Output only a "
    "single ```python code block containing the asserts, no imports of the "
    "module under test, no test framework, no commentary."
)
REFLECTION_SYSTEM = (
    "You are analysing your own failed attempt. In a few sentences, state why "
    "the implementation failed and what to do differently. Write no code."
)

MAX_ITERS = 2
MEMORY_OMEGA = 1
EXEC_TIMEOUT_S = 10
EXEC_MEMORY_BYTES = 2 * 1024**3

_CODE_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def extract_code(reply: str) -> str:
    """Returns the first fenced code block, or the whole reply if unfenced."""
    match = _CODE_BLOCK.search(reply)
    return (match.group(1) if match else reply).strip()


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (EXEC_TIMEOUT_S, EXEC_TIMEOUT_S))
    resource.setrlimit(resource.RLIMIT_AS, (EXEC_MEMORY_BYTES, EXEC_MEMORY_BYTES))


def run_tests(implementation: str, tests: str) -> tuple[bool, str]:
    """Executes an implementation against assert-style tests in a subprocess.

    Args:
        implementation: Source defining the function under test.
        tests: Statements that exercise it and raise on failure.

    Returns:
        Whether the run exited cleanly, and the captured failure text.
    """
    program = f"{implementation}\n\n{tests}\n"
    with tempfile.TemporaryDirectory(prefix="kvpim-t7-") as workdir:
        path = Path(workdir) / "candidate.py"
        path.write_text(program)
        try:
            done = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                timeout=EXEC_TIMEOUT_S,
                cwd=workdir,
                preexec_fn=_limits,
                env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return False, f"timed out after {EXEC_TIMEOUT_S}s"
        except Exception as error:  # noqa: BLE001 - report, never abort the run
            return False, f"{type(error).__name__}: {error}"
    if done.returncode == 0:
        return True, ""
    return False, (done.stderr or done.stdout or "non-zero exit").strip()[-1500:]


def build(task: dict, max_iters: int = MAX_ITERS) -> Iterator[Call]:
    """Yields the actor/evaluator/reflection stream for one HumanEval problem."""
    signature = f"```python\n{task['prompt']}```"
    reflections: list[str] = []
    implementation = None

    for iteration in range(max_iters):
        actor_context = f"Implement this function.\n\n{signature}"
        if implementation is not None:
            actor_context += f"\n\n## your previous attempt\n{implementation}"
        if reflections:
            actor_context += "\n\n## reflection\n" + "\n".join(reflections)

        implementation = yield Call(
            agent_id="actor",
            parent_idx=None,
            messages=[
                {"role": "system", "content": ACTOR_SYSTEM},
                {"role": "user", "content": actor_context},
            ],
            meta={"stage": "actor", "iteration": iteration},
        )

        unit_tests = yield Call(
            agent_id="evaluator",
            parent_idx=None,
            messages=[
                {"role": "system", "content": EVALUATOR_SYSTEM},
                {
                    "role": "user",
                    "content": f"{signature}\n\n## implementation\n{implementation}",
                },
            ],
            meta={"stage": "evaluator", "iteration": iteration},
        )

        passed, failure = run_tests(
            extract_code(implementation), extract_code(unit_tests)
        )
        if passed or iteration == max_iters - 1:
            break

        reflection = yield Call(
            agent_id="self_reflection",
            parent_idx=None,
            messages=[
                {"role": "system", "content": REFLECTION_SYSTEM},
                {
                    "role": "user",
                    "content": f"{signature}\n\n## implementation\n{implementation}"
                    f"\n\n## test failure\n{failure}",
                },
            ],
            meta={"stage": "reflection", "iteration": iteration, "failure": failure[:300]},
        )
        # Omega = 1: only the newest reflection survives into the next actor call.
        reflections = [reflection][-MEMORY_OMEGA:]


BASELINE_PATH = Path("/home/cw636/chenyi/KVPIM/traces/_baseline/humaneval_baseline.json")


def load_tasks(num_tasks: int = 10, seed: int = 0) -> list[dict]:
    """Draws problems the baseline fails on its first attempt.

    A problem the actor solves immediately yields no iteration trace, so
    Reflexion evaluates on the failures (plan section 3). The failure set comes
    from the baseline pass; without it, difficulty is proxied by canonical
    solution length and the hardest problems are drawn instead.
    """
    import datasets

    split = datasets.load_dataset("openai/openai_humaneval", split="test")
    failed = set()
    if BASELINE_PATH.exists():
        failed = set(json.loads(BASELINE_PATH.read_text())["failed_task_ids"])

    if len(failed) >= num_tasks:
        pool = [i for i in range(len(split)) if split[i]["task_id"] in failed]
    else:
        pool = sorted(
            range(len(split)),
            key=lambda i: len(split[i]["canonical_solution"].splitlines()),
            reverse=True,
        )[: num_tasks * 3]
    picks = random.Random(seed).sample(pool, num_tasks)
    return [
        {
            "task_id": split[i]["task_id"],
            "prompt": split[i]["prompt"],
            "test": split[i]["test"],
            "entry_point": split[i]["entry_point"],
            "solution_lines": len(split[i]["canonical_solution"].splitlines()),
        }
        for i in sorted(picks)
    ]
