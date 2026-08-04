# SPDX-License-Identifier: Apache-2.0
"""The single LLM call every topology driver emits."""

from dataclasses import dataclass, field


@dataclass
class Call:
    """One LLM invocation in a multi-agent workflow.

    Topology drivers yield these; the runner fills in the fields it owns
    (`topology`, `workflow_id`, `call_idx`) and sends back the reply text.

    Attributes:
        agent_id: Role name, i.e. the identity carried by the system prompt.
        parent_idx: DAG edge to the call this one consumes output from.
        messages: Chat messages, tokenized by the runner via the chat template.
    """

    agent_id: str
    messages: list[dict]
    parent_idx: int | None = None
    topology: str = ""
    workflow_id: str = ""
    call_idx: int = -1
    meta: dict = field(default_factory=dict)
