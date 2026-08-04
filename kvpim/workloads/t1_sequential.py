# SPDX-License-Identifier: Apache-2.0
"""T1 Sequential / Pipeline on HumanEval, following MetaGPT's waterfall.

Five roles hand work down a line — Product Manager, Architect, Project Manager,
Engineer, QaEngineer — with the context growing monotonically while the system
prompt is swapped at every step (plan section 3). QA can send the work back to
the Engineer, bounded by `MAX_RETRIES`, which is MetaGPT's executable-feedback
loop; the retry is decided by actually running the tests.

The role system prompts are assembled exactly as MetaGPT assembles them, from
`ref/MetaGPT/metagpt/roles/role.py`:

    PREFIX_TEMPLATE      "You are a {profile}, named {name}, your goal is {goal}. "
    CONSTRAINT_TEMPLATE  "the constraint is {constraints}. "

with each role's own `name`/`profile`/`goal`/`constraints` quoted verbatim from
`metagpt/roles/*.py`. `check_metagpt_reference` in `kvpim.analyze` re-derives
them from the repository and fails if they ever drift.

One deviation remains: MetaGPT drives the pipeline through structured artifacts
(a PRD document, a system design, a task list) produced by its Action classes.
Here each role is prompted directly with the accumulated context, so the
artifacts are free-form rather than schema-bound.
"""

import random
from collections.abc import Iterator

from kvpim.call import Call
from kvpim.workloads.t7_reflection import extract_code, run_tests

# ref/MetaGPT/metagpt/roles/role.py:51-52, verbatim.
PREFIX_TEMPLATE = "You are a {profile}, named {name}, your goal is {goal}. "
CONSTRAINT_TEMPLATE = "the constraint is {constraints}. "

# ref/MetaGPT/metagpt/roles/{product_manager,architect,project_manager,
# engineer,qa_engineer}.py — name/profile/goal/constraints, verbatim.
ROLES = {
    "product_manager": {
        "name": "Alice",
        "profile": "Product Manager",
        "goal": "Create a Product Requirement Document or market research/competitive product research.",  # noqa: E501
        "constraints": "utilize the same language as the user requirements for seamless communication",  # noqa: E501
    },
    "architect": {
        "name": "Bob",
        "profile": "Architect",
        "goal": "design a concise, usable, complete software system. output the system design.",  # noqa: E501
        "constraints": "make sure the architecture is simple enough and use  appropriate open source libraries. Use same language as user requirement",  # noqa: E501
    },
    "project_manager": {
        "name": "Eve",
        "profile": "Project Manager",
        "goal": "break down tasks according to PRD/technical design, generate a task list, and analyze task dependencies to start with the prerequisite modules",  # noqa: E501
        "constraints": "use same language as user requirement",
    },
    "engineer": {
        "name": "Alex",
        "profile": "Engineer",
        "goal": "write elegant, readable, extensible, efficient code",
        "constraints": "the code should conform to standards like google-style and be modular and maintainable. Use same language as user requirement",  # noqa: E501
    },
    "qa_engineer": {
        "name": "Edward",
        "profile": "QaEngineer",
        "goal": "Write comprehensive and robust tests to ensure codes will work as expected without bugs",  # noqa: E501
        "constraints": "The test code you write should conform to code standard like PEP8, be modular, easy to read and maintain.Use same language as user requirement",  # noqa: E501
    },
}

PIPELINE = ["product_manager", "architect", "project_manager", "engineer"]
MAX_RETRIES = 3


def role_system_prompt(role: str) -> str:
    """Builds a role's system prompt the way `Role._get_prefix` does."""
    fields = ROLES[role]
    prefix = PREFIX_TEMPLATE.format(**{k: fields[k] for k in ("profile", "name", "goal")})
    if fields["constraints"]:
        prefix += CONSTRAINT_TEMPLATE.format(constraints=fields["constraints"])
    return prefix


def build(task: dict) -> Iterator[Call]:
    """Yields the waterfall call stream for one HumanEval problem."""
    context = (
        f"Implement the following Python function.\n\n```python\n{task['prompt']}```"
    )

    for step, role in enumerate(PIPELINE):
        reply = yield Call(
            agent_id=role,
            parent_idx=step - 1 if step else None,
            messages=[
                {"role": "system", "content": role_system_prompt(role)},
                {"role": "user", "content": context},
            ],
            meta={"stage": role, "attempt": 0},
        )
        context += f"\n\n## {role}\n{reply}"
        if role == "engineer":
            implementation = reply

    engineer_idx = len(PIPELINE) - 1
    for attempt in range(MAX_RETRIES + 1):
        tests = yield Call(
            agent_id="qa_engineer",
            parent_idx=engineer_idx,
            messages=[
                {"role": "system", "content": role_system_prompt("qa_engineer")},
                {
                    "role": "user",
                    "content": f"{context}\n\nWrite tests for this implementation as a "
                    "series of bare `assert` statements calling the function "
                    "directly. Output only a single ```python code block.",
                },
            ],
            meta={"stage": "qa_engineer", "attempt": attempt},
        )
        passed, failure = run_tests(extract_code(implementation), extract_code(tests))
        context += f"\n\n## qa_engineer (attempt {attempt})\n{tests}"
        if passed:
            break
        context += f"\n\n## test failure\n{failure}"
        if attempt == MAX_RETRIES:
            break

        engineer_idx = None
        implementation = yield Call(
            agent_id="engineer",
            parent_idx=None,
            messages=[
                {"role": "system", "content": role_system_prompt("engineer")},
                {
                    "role": "user",
                    "content": f"{context}\n\nFix the implementation and output "
                    "only the corrected ```python code block.",
                },
            ],
            meta={"stage": "engineer_retry", "attempt": attempt + 1,
                  "failure": failure[:300]},
        )
        context += f"\n\n## engineer (retry {attempt + 1})\n{implementation}"


def load_tasks(num_tasks: int = 10, seed: int = 0) -> list[dict]:
    """Samples HumanEval problems, stratified by canonical solution length.

    Short/medium/long strata keep the pipeline's context growth representative
    rather than clustering on trivial problems (plan section 3).
    """
    import datasets

    split = datasets.load_dataset("openai/openai_humaneval", split="test")
    order = sorted(
        range(len(split)), key=lambda i: len(split[i]["canonical_solution"].splitlines())
    )
    rng = random.Random(seed)
    bucket = len(order) / num_tasks
    picks = [
        order[rng.randrange(int(b * bucket), int((b + 1) * bucket))]
        for b in range(num_tasks)
    ]
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
