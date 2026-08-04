# SPDX-License-Identifier: Apache-2.0
"""T5 Group Chat on MATH level 5, following AutoGen's dynamic GroupChat.

Three agents (User_proxy, Solver, Checker) share one broadcast transcript while
a manager picks the next speaker with its own LLM call every round — that
selection call is part of the trace, as the plan requires. `max_round=12`
bounds the session.

Speaker-selection prompts are quoted verbatim from AutoGen 0.2.34's
`autogen/agentchat/groupchat.py` (`select_speaker_message_template` /
`select_speaker_prompt_template`); the 0.4 line in `ref/autogen` no longer ships
GroupChat, so the 0.2.34 tag is fetched into the same clone.

The transcript is shared but each speaker's own system prompt sits in front of
it, so every speaker's prefix forks at token zero — the "SYS in front" default
the plan keeps for the main experiment (plan section 2 of KVPIM-README).
"""

import random
import re
from collections.abc import Iterator

from kvpim.call import Call

# autogen v0.2.34 autogen/agentchat/groupchat.py, verbatim.
SELECT_SPEAKER_MESSAGE_TEMPLATE = """You are in a role play game. The following roles are available:
                {roles}.
                Read the following conversation.
                Then select the next role from {agentlist} to play. Only return the role."""  # noqa: E501
SELECT_SPEAKER_PROMPT_TEMPLATE = (
    "Read the above conversation. Then select the next role from {agentlist} to "
    "play. Only return the role."
)

AGENTS = {
    "User_proxy": "You are the user proxy. Restate what is still needed to finish "
    "the problem, and confirm when the solution is complete.",
    "Solver": "You are the solver. Work the mathematics step by step and state the "
    "final answer in \\boxed{}.",
    "Checker": "You are the checker. Verify the solver's reasoning and arithmetic, "
    "and say explicitly whether the boxed answer is correct.",
}
MAX_ROUND = 12


def _roles_block() -> str:
    return "\n".join(f"{name}: {system}" for name, system in AGENTS.items())


def parse_speaker(reply: str, names: list[str]) -> str:
    """Resolves the manager's reply to one of the agent names."""
    lowered = reply.lower()
    hits = [(lowered.find(name.lower()), name) for name in names if name.lower() in lowered]
    return min(hits)[1] if hits else names[0]


def build(task: dict, max_round: int = MAX_ROUND) -> Iterator[Call]:
    """Yields the manager/speaker call stream for one MATH problem."""
    names = list(AGENTS)
    agentlist = "[" + ", ".join(names) + "]"
    transcript = [{"role": "user", "content": task["problem"]}]

    for turn in range(max_round):
        # The manager's speaker-selection call is itself part of the trace.
        selection = yield Call(
            agent_id="manager",
            parent_idx=2 * turn - 1 if turn else None,
            messages=[
                {
                    "role": "system",
                    "content": SELECT_SPEAKER_MESSAGE_TEMPLATE.format(
                        roles=_roles_block(), agentlist=agentlist
                    ),
                },
                *transcript,
                {
                    "role": "user",
                    "content": SELECT_SPEAKER_PROMPT_TEMPLATE.format(
                        agentlist=agentlist
                    ),
                },
            ],
            meta={"stage": "select_speaker", "turn": turn},
        )
        speaker = parse_speaker(selection, names)

        reply = yield Call(
            agent_id=speaker,
            parent_idx=2 * turn,
            messages=[{"role": "system", "content": AGENTS[speaker]}, *transcript],
            meta={"stage": "speak", "turn": turn, "speaker": speaker},
        )
        transcript.append({"role": "assistant", "content": f"{speaker}: {reply}"})


_LEVEL5 = re.compile(r"Level 5")


def load_tasks(num_tasks: int = 10, seed: int = 0) -> list[dict]:
    """Samples MATH level-5 problems, spread over the non-geometry subjects.

    AutoGen's scenario A5 draws level-5 problems from the six subjects other
    than geometry, whose figures the text-only setting cannot carry.
    """
    import datasets

    split = datasets.load_dataset("EleutherAI/hendrycks_math", "algebra", split="test")
    subjects = [
        "algebra",
        "counting_and_probability",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    rng = random.Random(seed)
    picked = []
    for i, subject in enumerate(subjects):
        if len(picked) >= num_tasks:
            break
        split = datasets.load_dataset("EleutherAI/hendrycks_math", subject, split="test")
        level5 = [r for r in split if _LEVEL5.search(r.get("level", ""))]
        share = 2 if i < num_tasks - len(subjects) else 1
        picked.extend(
            {
                "task_id": f"math_{subject}_{j}",
                "problem": row["problem"],
                "solution": row["solution"],
                "subject": subject,
            }
            for j, row in enumerate(rng.sample(level5, min(share, len(level5))))
        )
    return picked[:num_tasks]
