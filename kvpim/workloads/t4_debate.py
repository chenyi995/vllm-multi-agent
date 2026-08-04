# SPDX-License-Identifier: Apache-2.0
"""T4 Debate on GSM8K, following Du et al. (ICML'24).

Three agents answer independently in round 0; in every later round each one is
shown the other agents' full replies and answers again. Prompts are quoted
verbatim from `ref/llm_multiagent_debate/gsm/gen_gsm.py`, including the paper's
3 agents / 2 rounds and its `random.seed(0)` question sampling.

This is the position-mismatch battleground of the matrix: agent A and agent B
see the same peer replies concatenated in a different order, so identical text
lands at different offsets and `K_post` cannot match even where `K_derope` can
(plan section 3).
"""

import json
import random
from collections.abc import Iterator

from kvpim.call import Call

# ref/llm_multiagent_debate/gsm/gen_gsm.py, verbatim.
QUESTION_PROMPT = """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """  # noqa: E501
DEBATE_PREFIX = "These are the solutions to the problem from other agents: "
AGENT_SOLUTION = "\n\n One agent solution: ```{}```"
DEBATE_SUFFIX = """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."""  # noqa: E501

NUM_AGENTS = 3
NUM_ROUNDS = 2


def construct_message(other_replies: list[str], question: str) -> str:
    """Mirrors `construct_message` for the non-empty case, verbatim."""
    prefix_string = DEBATE_PREFIX
    for reply in other_replies:
        prefix_string = prefix_string + AGENT_SOLUTION.format(reply)
    return prefix_string + DEBATE_SUFFIX.format(question)


def build(
    task: dict, num_agents: int = NUM_AGENTS, num_rounds: int = NUM_ROUNDS
) -> Iterator[Call]:
    """Yields the debate call stream for one GSM8K problem.

    Each agent carries its own conversation, so the peer replies arrive in an
    order that differs per agent — deliberately, that is what the topology is.
    """
    question = task["question"]
    # Per-agent conversation, exactly as the reference keeps `agent_contexts`.
    contexts = [
        [{"role": "user", "content": QUESTION_PROMPT.format(question)}]
        for _ in range(num_agents)
    ]
    last_round: list[str] = []

    for round_idx in range(num_rounds):
        replies = []
        for agent in range(num_agents):
            if round_idx:
                others = [r for i, r in enumerate(last_round) if i != agent]
                contexts[agent].append(
                    {"role": "user", "content": construct_message(others, question)}
                )
            reply = yield Call(
                agent_id=f"agent_{agent}",
                parent_idx=(round_idx - 1) * num_agents if round_idx else None,
                messages=list(contexts[agent]),
                meta={"round": round_idx, "agent": agent},
            )
            contexts[agent].append({"role": "assistant", "content": reply})
            replies.append(reply)
        last_round = replies


def load_tasks(num_tasks: int = 10, seed: int = 0) -> list[dict]:
    """Samples GSM8K test problems the way the reference shuffles them."""
    import datasets

    split = datasets.load_dataset("openai/gsm8k", "main", split="test")
    order = list(range(len(split)))
    random.Random(seed).shuffle(order)
    return [
        {
            "task_id": f"gsm8k_{i}",
            "question": split[i]["question"],
            "answer": split[i]["answer"],
        }
        for i in order[:num_tasks]
    ]
