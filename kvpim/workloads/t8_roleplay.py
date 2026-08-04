# SPDX-License-Identifier: Apache-2.0
"""T8 Role-play / Simulation on CAMEL AI Society.

Inception Prompting drives two agents: the AI User only issues
"Instruction:/Input:" and the AI Assistant only answers with "Solution:". Both
system prompts are quoted verbatim from `ref/camel/camel/prompts/ai_society.py`
(ASSISTANT_PROMPT / USER_PROMPT).

Each turn both agents see the whole transcript but under their own system
prompt, so the shared history forks at the very first token — this is the
topology with the longest reuse distance in the matrix (plan section 5.3).

Termination follows the paper: the `<CAMEL_TASK_DONE>` token, or the hard cap of
40 messages, which is what keeps the working set bounded (plan section 3).
"""

import random
import re
from collections.abc import Iterator

from kvpim.call import Call

# ref/camel/camel/prompts/ai_society.py::ASSISTANT_PROMPT, quoted exactly.
ASSISTANT_PROMPT = """===== RULES OF ASSISTANT =====
Never forget you are a {assistant_role} and I am a {user_role}. Never flip roles! Never instruct me!
We share a common interest in collaborating to successfully complete a task.
You must help me to complete the task.
Here is the task: {task}. Never forget our task!
I must instruct you based on your expertise and my needs to complete the task.

I must give you one instruction at a time.
You must write a specific solution that appropriately solves the requested instruction and explain your solutions.
You must decline my instruction honestly if you cannot perform the instruction due to physical, moral, legal reasons or your capability and explain the reasons.
Unless I say the task is completed, you should always start with:

Solution: <YOUR_SOLUTION>

<YOUR_SOLUTION> should be very specific, include detailed explanations and provide preferable detailed implementations and examples and lists for task-solving.
Always end <YOUR_SOLUTION> with: Next request."""  # noqa: E501

# ref/camel/camel/prompts/ai_society.py::USER_PROMPT, quoted exactly.
USER_PROMPT = """===== RULES OF USER =====
Never forget you are a {user_role} and I am a {assistant_role}. Never flip roles! You will always instruct me.
We share a common interest in collaborating to successfully complete a task.
I must help you to complete the task.
Here is the task: {task}. Never forget our task!
You must instruct me based on my expertise and your needs to solve the task ONLY in the following two ways:

1. Instruct with a necessary input:
Instruction: <YOUR_INSTRUCTION>
Input: <YOUR_INPUT>

2. Instruct without any input:
Instruction: <YOUR_INSTRUCTION>
Input: None

The "Instruction" describes a task or question. The paired "Input" provides further context or information for the requested "Instruction".

You must give me one instruction at a time.
I must write a response that appropriately solves the requested instruction.
I must decline your instruction honestly if I cannot perform the instruction due to physical, moral, legal reasons or my capability and explain the reasons.
You should instruct me not ask me questions.
Now you must start to instruct me using the two ways described above.
Do not add anything else other than your instruction and the optional corresponding input!
Keep giving me instructions and necessary inputs until you think the task is completed.
When the task is completed, you must only reply with a single word <CAMEL_TASK_DONE>.
Never say <CAMEL_TASK_DONE> unless my responses have solved your task."""  # noqa: E501

MAX_MESSAGES = 40
TASK_DONE = "<CAMEL_TASK_DONE>"
_ROLE_SUFFIX = re.compile(r"_RoleType\.(ASSISTANT|USER)$")


def _role_name(raw: str) -> str:
    return _ROLE_SUFFIX.sub("", raw).replace("_", " ")


def build(task: dict, max_messages: int = MAX_MESSAGES) -> Iterator[Call]:
    """Yields the alternating user/assistant stream for one role-play session."""
    fields = {
        "assistant_role": task["assistant_role"],
        "user_role": task["user_role"],
        "task": task["specified_task"],
    }
    user_system = USER_PROMPT.format(**fields)
    assistant_system = ASSISTANT_PROMPT.format(**fields)

    # CAMEL kicks the session off by asking the user agent to begin; that turn
    # is part of the transcript both agents carry from then on.
    transcript: list[dict] = [
        {"role": "user", "content": "Now start to give me instructions."}
    ]
    for turn in range(max_messages // 2):
        instruction = yield Call(
            agent_id="ai_user",
            parent_idx=2 * turn - 1 if turn else None,
            messages=[{"role": "system", "content": user_system}, *transcript],
            meta={"stage": "ai_user", "turn": turn},
        )
        transcript.append({"role": "assistant", "content": instruction})
        if TASK_DONE in instruction:
            break

        solution = yield Call(
            agent_id="ai_assistant",
            parent_idx=2 * turn,
            messages=[
                {"role": "system", "content": assistant_system},
                # The assistant reads the same transcript with the roles mirrored.
                *[
                    {"role": "user" if m["role"] == "assistant" else "assistant",
                     "content": m["content"]}
                    for m in transcript
                ],
            ],
            meta={"stage": "ai_assistant", "turn": turn},
        )
        transcript.append({"role": "user", "content": solution})


def load_tasks(num_tasks: int = 10, seed: int = 0) -> list[dict]:
    """Samples role pairs from CAMEL AI Society, stratified by assistant role."""
    import json

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "camel-ai/ai_society", "ai_society_instructions.json", repo_type="dataset"
    )
    records = json.load(open(path))

    by_pair: dict[tuple, dict] = {}
    for record in records:
        key = (record["role_1"], record["role_2"], record["specified_task"])
        by_pair.setdefault(key, record)

    rng = random.Random(seed)
    keys = sorted(by_pair)
    by_assistant: dict[str, list] = {}
    for key in keys:
        by_assistant.setdefault(key[0], []).append(key)

    picked = []
    roles = sorted(by_assistant)
    rng.shuffle(roles)
    for role in roles:
        if len(picked) >= num_tasks:
            break
        picked.append(rng.choice(by_assistant[role]))

    return [
        {
            "task_id": by_pair[key]["id"],
            "assistant_role": _role_name(key[0]),
            "user_role": _role_name(key[1]),
            "specified_task": key[2],
        }
        for key in picked
    ]
