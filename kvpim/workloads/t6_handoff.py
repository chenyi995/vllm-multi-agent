# SPDX-License-Identifier: Apache-2.0
"""T6 Handoff / Swarm on tau-bench's airline domain.

Five agents hand control down a hierarchy — Triage, Flight Modification, Flight
Cancel, Flight Change, Lost Baggage. Swarm's semantics are that a handoff does
**not** trim the conversation: only the system prompt is swapped (README), so
the prefix breaks at the very front while the whole history shifts by the
difference in system prompt lengths. That shift is why this topology matters for
`K_derope` (plan section 3).

tau-bench supplies both the task instructions and an LLM user simulator whose
calls are themselves part of the trace, terminating on `###STOP###`.

Prompts are read out of `ref/` at load time rather than embedded: the Swarm
agent instructions are `STARTER_PROMPT` plus a domain policy, roughly 6 KB in
total, and copying them here would create a second copy to drift from. The
Tier 1 guarantee is therefore structural — the strings *are* the reference.
"""

import random
import re
from collections.abc import Iterator
from pathlib import Path

from kvpim.call import Call

REF_ROOT = Path(__file__).resolve().parents[2] / "ref"
AIRLINE = REF_ROOT / "swarm/examples/airline"
TAU_BENCH = REF_ROOT / "tau-bench/tau_bench/envs"

MAX_TURNS = 30
STOP_TOKEN = "###STOP###"


def _constants(path: Path, names: set[str]) -> dict[str, str]:
    """Reads string constants out of a Swarm prompt module.

    The policy files are plain string tables with no imports, but some are
    f-strings, so they are executed rather than read off the syntax tree.
    """
    source = path.read_text()
    assert not any(
        line.startswith(("import ", "from ")) for line in source.splitlines()
    ), f"{path} unexpectedly imports; refusing to execute it"
    namespace: dict = {}
    exec(compile(source, str(path), "exec"), namespace)
    found = {name: namespace[name] for name in names if isinstance(namespace.get(name), str)}
    missing = names - found.keys()
    if missing:
        raise LookupError(f"{sorted(missing)} not found in {path}")
    return found


def load_agents() -> dict[str, str]:
    """Builds the five agents' system prompts from Swarm's own sources."""
    starter = _constants(AIRLINE / "data/routines/prompts.py", {"STARTER_PROMPT"})
    triage = _constants(
        AIRLINE / "data/routines/prompts.py", {"TRIAGE_SYSTEM_PROMPT"}
    )["TRIAGE_SYSTEM_PROMPT"]
    modification = _constants(
        AIRLINE / "data/routines/flight_modification/policies.py",
        {"FLIGHT_CANCELLATION_POLICY", "FLIGHT_CHANGE_POLICY"},
    )
    baggage = _constants(
        AIRLINE / "data/routines/baggage/policies.py", {"LOST_BAGGAGE_POLICY"}
    )["LOST_BAGGAGE_POLICY"]
    prefix = starter["STARTER_PROMPT"]
    return {
        "triage": triage,
        "flight_modification": (
            "You are a Flight Modification Agent for a customer service airlines "
            "company.\n      You are an expert customer service agent deciding which "
            "sub intent the user should be referred to.\nYou already know the intent "
            "is for flight modification related question. First, look at message "
            "history and see if you can determine if the user wants to cancel or "
            "change their flight.\nAsk user clarifying questions until you know "
            "whether or not it is a cancel request or change flight request. Once you "
            "know, call the appropriate transfer function. Either ask clarifying "
            "questions, or call one of your functions, every time."
        ),
        "flight_cancel": prefix + modification["FLIGHT_CANCELLATION_POLICY"],
        "flight_change": prefix + modification["FLIGHT_CHANGE_POLICY"],
        "lost_baggage": prefix + baggage,
    }


# The handoff graph Swarm wires through its transfer_to_* functions.
HANDOFFS = {
    "triage": {"flight_modification", "lost_baggage"},
    "flight_modification": {"flight_cancel", "flight_change"},
    "flight_cancel": {"triage"},
    "flight_change": {"triage"},
    "lost_baggage": {"triage"},
}
_AGENT_MENTION = {
    "flight_modification": ("modif", "change or cancel"),
    "flight_cancel": ("cancel",),
    "flight_change": ("change", "rebook"),
    "lost_baggage": ("baggage", "luggage"),
    "triage": ("triage",),
}


def next_agent(current: str, reply: str) -> str:
    """Picks the handoff target the reply asks for, if it is a legal edge."""
    lowered = reply.lower()
    for candidate in HANDOFFS[current]:
        if any(word in lowered for word in _AGENT_MENTION[candidate]):
            return candidate
    return current


def build(task: dict, max_turns: int = MAX_TURNS) -> Iterator[Call]:
    """Yields the alternating simulated-user / airline-agent stream."""
    agents = load_agents()
    user_system = USER_SIMULATOR_PROMPT.format(instruction=task["instruction"])

    current = "triage"
    # Swarm never trims: this history is carried across every handoff.
    history: list[dict] = []

    for turn in range(max_turns // 2):
        user_message = yield Call(
            agent_id="user_simulator",
            parent_idx=2 * turn - 1 if turn else None,
            messages=[
                {"role": "system", "content": user_system},
                *[
                    {"role": "user" if m["role"] == "assistant" else "assistant",
                     "content": m["content"]}
                    for m in history
                ],
            ]
            if history
            else [
                {"role": "system", "content": user_system},
                {"role": "user", "content": "Hi, how can I help you today?"},
            ],
            meta={"stage": "user", "turn": turn, "agent": current},
        )
        if STOP_TOKEN in user_message:
            break
        history.append({"role": "user", "content": user_message})

        reply = yield Call(
            agent_id=current,
            parent_idx=2 * turn,
            # Only the system prompt changes on a handoff; the history is intact
            # and every token in it shifts by the system prompts' length delta.
            messages=[{"role": "system", "content": agents[current]}, *history],
            meta={"stage": "agent", "turn": turn, "agent": current},
        )
        history.append({"role": "assistant", "content": reply})
        current = next_agent(current, reply)


# ref/tau-bench/tau_bench/envs/user.py, the non-reasoning simulator prompt.
USER_SIMULATOR_PROMPT = """You are a user interacting with an agent.

Instruction: {instruction}

Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""  # noqa: E501

_INSTRUCTION = re.compile(r'instruction="((?:[^"\\]|\\.)*)"', re.DOTALL)
_USER_ID = re.compile(r'user_id="([^"]*)"')


def load_tasks(num_tasks: int = 10, seed: int = 0) -> list[dict]:
    """Samples tau-bench airline tasks, evenly spaced over the 50 test tasks."""
    source = (TAU_BENCH / "airline/tasks_test.py").read_text()
    instructions = [m.group(1) for m in _INSTRUCTION.finditer(source)]
    user_ids = [m.group(1) for m in _USER_ID.finditer(source)]
    del seed  # the slice is deterministic; nothing is drawn at random
    step = max(len(instructions) // num_tasks, 1)
    picks = list(range(0, len(instructions), step))[:num_tasks]
    return [
        {
            "task_id": f"tau_airline_{i}",
            "instruction": instructions[i].encode().decode("unicode_escape"),
            "user_id": user_ids[i] if i < len(user_ids) else "",
        }
        for i in picks
    ]
