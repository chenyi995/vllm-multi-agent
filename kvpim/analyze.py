# SPDX-License-Identifier: Apache-2.0
"""Offline analysis over a trace directory.

Everything downstream trusts `blocks.jsonl`, so the first thing this module can
do is prove the event stream is complete: rebuild the radix tree from the stored
blocks' token ids and check the prefix hit it predicts against the engine's own
`num_cached_tokens` (plan section 9, sanity #4).

Note on what a `BlockStored` event means here: vLLM only re-emits events for
prefix-cache *reuses* when a request sets `kv_cache_report_mode="full"`, which is
not the default. Under the default the stream carries newly cached blocks only,
which is exactly the population `N_total` counts.
"""

import ast
import copy
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

REF_ROOT = Path(__file__).resolve().parents[1] / "ref"


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class RadixTree:
    """The cached-prefix trie as rebuilt from `BlockStored` events."""

    children: dict[object, list[tuple[object, tuple[int, ...]]]]

    @classmethod
    def from_events(cls, events: list[dict], until_ts: float | None = None):
        """Replays the stream up to ``until_ts`` into the tree it implies.

        A reset (`AllBlocksCleared`, emitted between workflows) empties the
        cache and an eviction (`BlockRemoved`) drops individual blocks; both must
        be applied, or the tree keeps blocks the engine can no longer hit and
        over-predicts the next call's prefix.
        """
        children: dict[object, list[tuple[object, tuple[int, ...]]]] = {}
        for event in events:
            if until_ts is not None and event["ts"] > until_ts:
                continue
            if event["type"] == "AllBlocksCleared":
                children = {}
                continue
            if event["type"] == "BlockRemoved":
                dropped = set(event["block_hashes"])
                children = {
                    parent: [c for c in kids if c[0] not in dropped]
                    for parent, kids in children.items()
                    if parent not in dropped
                }
                continue
            if event["type"] != "BlockStored":
                continue
            parent = event["parent_block_hash"]
            token_ids = event["token_ids"]
            size = event["block_size"]
            for i, block_hash in enumerate(event["block_hashes"]):
                chunk = tuple(token_ids[i * size : (i + 1) * size])
                children.setdefault(parent, []).append((block_hash, chunk))
                parent = block_hash
        return cls(children)

    def longest_prefix(self, token_ids: list[int]) -> int:
        """Tokens of ``token_ids`` already covered by a cached prefix chain."""
        parent, matched = None, 0
        while True:
            for block_hash, chunk in self.children.get(parent, ()):
                if tuple(token_ids[matched : matched + len(chunk)]) == chunk:
                    matched += len(chunk)
                    parent = block_hash
                    break
            else:
                return matched


def reconcile_events(trace_dir: str | Path) -> dict:
    """Sanity #4: engine-reported hits must match the rebuilt radix tree.

    The engine never reports the whole prompt as cached — it keeps at least the
    final block to recompute — so the tree may legitimately predict up to one
    block more than `num_cached_tokens`.

    Args:
        trace_dir: A configuration directory holding calls/blocks/manifest.

    Returns:
        A summary with the per-call mismatches, if any.
    """
    trace_dir = Path(trace_dir)
    manifest = json.loads((trace_dir / "manifest.json").read_text())
    calls = read_jsonl(trace_dir / "calls.jsonl")
    events = read_jsonl(trace_dir / "blocks.jsonl")
    # The engine always leaves the final *physical* block to recompute, even when
    # the matching unit is finer, so the allowance is the block size — not
    # `prefix_match_unit`.
    unit = manifest["block_size"]

    # The cache is cleared between workflows, and the resulting AllBlocksCleared
    # is timestamped by the engine's publisher thread, which can land *after*
    # the driver's own t_start for the next call. Filtering on time alone would
    # then carry the previous workflow's blocks over and over-predict the hit,
    # so the stream is segmented on those markers instead and each workflow is
    # reconciled against its own segment.
    segments: list[list[dict]] = []
    current: list[dict] = []
    for event in events:
        if event["type"] == "AllBlocksCleared":
            if current:
                segments.append(current)
            current = []
        else:
            current.append(event)
    if current:
        segments.append(current)

    workflow_order: list[str] = []
    for call in calls:
        if call["workflow_id"] not in workflow_order:
            workflow_order.append(call["workflow_id"])
    # Publication lag can also push a workflow's last events past the next
    # workflow's clear marker, so drop anything timestamped before the workflow
    # actually started: after a reset the cache is empty by construction, and
    # such events can only be stragglers the engine no longer holds.
    first_start = {}
    for call in calls:
        first_start.setdefault(call["workflow_id"], call["t_start"])
    per_workflow = {
        workflow_id: [
            event
            for event in (segments[i] if i < len(segments) else [])
            if event["ts"] >= first_start[workflow_id]
        ]
        for i, workflow_id in enumerate(workflow_order)
    }

    # Two modes, because the two capacity tiers support different claims.
    #
    #   ample   no eviction, so the cache state at any instant can be replayed
    #           exactly: require predicted == reported (the engine keeps the
    #           final physical block, hence the one-block allowance).
    #
    #   limited eviction churns constantly (one T4 workflow evicted 325 blocks)
    #           and the events are timestamped by the publisher thread, so the
    #           exact state at an instant is NOT reconstructible. What is still
    #           checkable — and is what sanity #4 exists for — is completeness:
    #           every token the engine claims to have hit must belong to a block
    #           this workflow actually stored. Evictions are ignored and the
    #           requirement is predicted >= reported.
    strict = manifest["capacity_tier"] == "ample"

    mismatches = []
    exact = 0
    for call in calls:
        if "t_start" not in call:
            raise ValueError("calls.jsonl predates t_start; re-run the config")
        workflow_events = per_workflow[call["workflow_id"]]
        if strict:
            tree = RadixTree.from_events(workflow_events, until_ts=call["t_start"])
        else:
            tree = RadixTree.from_events(
                [e for e in workflow_events if e["type"] == "BlockStored"]
            )
        predicted = tree.longest_prefix(call["prompt_token_ids"])
        reported = call["num_cached_tokens"]
        gap = predicted - reported
        if gap == 0:
            exact += 1
            continue
        bad = (not 0 < gap <= unit) if strict else gap < 0
        if bad:
            mismatches.append(
                {
                    "workflow_id": call["workflow_id"],
                    "call_idx": call["call_idx"],
                    "predicted": predicted,
                    "reported": reported,
                    "gap": gap,
                }
            )

    return {
        "trace_dir": str(trace_dir),
        "mode": "exact-replay" if strict else "completeness-only",
        "num_calls": len(calls),
        "exact_matches": exact,
        "within_tolerance": len(calls) - exact - len(mismatches),
        "mismatches": mismatches,
        "passed": not mismatches,
    }


# ---------------------------------------------------------------------------
# Topology correctness (plan section 9, sanity #5 replacement)
#
# The orchestration layer is the study's independent variable — vLLM cannot tell
# T1 from T9 — so a driver bug does not surface as an error, it surfaces as a
# finding. Two independent guards:
#   Tier 1  our prompt assembly must match the paper's own code, byte for byte.
#   Tier 2  the recorded token streams must satisfy the topology's structural
#           relations, stated from the topology definition rather than copied
#           from the driver.
# Neither catches a misreading of the topology itself; Tier 1 narrows that to
# whatever the reference implementation does not cover.
# ---------------------------------------------------------------------------


def load_reference_function(rel_path: str, func_name: str, globals_: dict | None = None):
    """Extracts one function from a `ref/` repo without importing the module.

    The reference files import heavyweight clients (`openai`, `requests`, ...)
    that are not installed here, so the function is compiled on its own from the
    repository's own source.

    Args:
        rel_path: Path of the source file relative to `ref/`.
        func_name: Function to extract.
        globals_: Names the function body needs.

    Returns:
        The compiled function object.
    """
    source = (REF_ROOT / rel_path).read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            namespace = dict(globals_ or {})
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, f"<{rel_path}:{func_name}>", "exec"), namespace)
            return namespace[func_name]
    raise LookupError(f"{func_name} not found in {rel_path}")


def check_moa_reference(num_references: int = 3) -> dict:
    """Tier 1 for T3: match MoA's own `inject_references_to_messages`.

    Compares both call sites the driver uses — a proposer, whose persona system
    message gets the references appended, and the aggregator, which has no
    system message and gets one created.

    Returns:
        Per-case equality against the reference implementation.
    """
    from kvpim.workloads.t3_fanout import PROPOSER_SYSTEM_PROMPTS, _with_references

    reference = load_reference_function(
        "MoA/utils.py", "inject_references_to_messages", {"copy": copy}
    )
    refs = [f"Reference answer number {i} with punctuation, and\nnewlines." for i in range(num_references)]
    task = "What are the names of some famous actors?"
    cases = []

    for i, persona in enumerate(PROPOSER_SYSTEM_PROMPTS):
        theirs = reference(
            [
                {"role": "system", "content": persona},
                {"role": "user", "content": task},
            ],
            refs,
        )
        ours = _with_references(persona, refs)
        cases.append(
            {
                "case": f"proposer_{i}",
                "equal": theirs[0]["content"] == ours,
                "ours_len": len(ours),
                "theirs_len": len(theirs[0]["content"]),
            }
        )

    theirs = reference([{"role": "user", "content": task}], refs)
    ours = _with_references(None, refs)
    cases.append(
        {
            "case": "aggregator",
            "equal": theirs[0]["role"] == "system" and theirs[0]["content"] == ours,
            "ours_len": len(ours),
            "theirs_len": len(theirs[0]["content"]),
        }
    )

    return {
        "check": "moa_inject_references",
        "reference": "ref/MoA/utils.py::inject_references_to_messages",
        "cases": cases,
        "passed": all(c["equal"] for c in cases),
    }


def check_metagpt_reference() -> dict:
    """Tier 1 for T1: role prompts must match what MetaGPT itself assembles.

    Re-reads `PREFIX_TEMPLATE` / `CONSTRAINT_TEMPLATE` from `role.py` and each
    role's own `name`/`profile`/`goal`/`constraints`, then rebuilds the system
    prompt and compares it with the driver's.

    Returns:
        Per-role equality against the reference repository.
    """
    from kvpim.workloads.t1_sequential import ROLES, role_system_prompt

    role_py = (REF_ROOT / "MetaGPT/metagpt/roles/role.py").read_text()
    templates = {}
    for node in ast.parse(role_py).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("PREFIX_TEMPLATE", "CONSTRAINT_TEMPLATE"):
                templates[node.targets[0].id] = ast.literal_eval(node.value)

    files = {
        "product_manager": "product_manager",
        "architect": "architect",
        "project_manager": "project_manager",
        "engineer": "engineer",
        "qa_engineer": "qa_engineer",
    }
    cases = []
    for role, filename in files.items():
        source = (REF_ROOT / f"MetaGPT/metagpt/roles/{filename}.py").read_text()
        fields = {}
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id in ("name", "profile", "goal", "constraints")
            ):
                try:
                    fields[node.target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
        theirs = templates["PREFIX_TEMPLATE"].format(
            profile=fields["profile"], name=fields["name"], goal=fields["goal"]
        )
        if fields.get("constraints"):
            theirs += templates["CONSTRAINT_TEMPLATE"].format(
                constraints=fields["constraints"]
            )
        ours = role_system_prompt(role)
        cases.append(
            {
                "case": role,
                "equal": theirs == ours,
                "fields_equal": all(
                    ROLES[role][k] == fields[k] for k in ("name", "profile", "goal")
                ),
                "length": len(ours),
            }
        )

    return {
        "check": "metagpt_role_prompts",
        "reference": "ref/MetaGPT/metagpt/roles/*.py",
        "cases": cases,
        "passed": all(c["equal"] for c in cases),
    }


def reference_source(rel_path: str, rev: str | None = None) -> str:
    """Returns the text of a `ref/` file, optionally at a given git revision.

    AutoGen's GroupChat was dropped in the 0.4 line that `ref/autogen` checks
    out, so its prompts only exist under the v0.2.34 tag fetched into the same
    clone.

    Args:
        rel_path: Path of the source file relative to `ref/`.
        rev: Git revision to read from; the working tree when None. The
            repository is taken to be the first path component.
    """
    if rev is None:
        return (REF_ROOT / rel_path).read_text()
    repo, _, inner = rel_path.partition("/")
    return subprocess.run(
        ["git", "-C", str(REF_ROOT / repo), "show", f"{rev}:{inner}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def load_reference_strings(
    rel_path: str, names: set[str], rev: str | None = None
) -> dict[str, str]:
    """Reads named string constants out of a `ref/` file without importing it.

    Handles both bare assignments and values wrapped in a single-argument call
    such as CAMEL's `TextPrompt("...")`, at module or class level.
    """
    source = reference_source(rel_path, rev)
    found: dict[str, str] = {}

    def record(target, value):
        if not isinstance(target, ast.Name) or target.id not in names:
            return
        if isinstance(value, ast.Call) and value.args:
            value = value.args[0]
        try:
            resolved = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return
        if isinstance(resolved, str):
            found[target.id] = resolved

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                record(target, node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            record(node.target, node.value)

    missing = names - found.keys()
    if missing:
        raise LookupError(f"{sorted(missing)} not found in {rel_path}")
    return found


def check_camel_reference() -> dict:
    """Tier 1 for T8: inception prompts must match CAMEL's own source."""
    from kvpim.workloads.t8_roleplay import ASSISTANT_PROMPT, USER_PROMPT

    reference = load_reference_strings(
        "camel/camel/prompts/ai_society.py", {"ASSISTANT_PROMPT", "USER_PROMPT"}
    )
    ours = {"ASSISTANT_PROMPT": ASSISTANT_PROMPT, "USER_PROMPT": USER_PROMPT}
    cases = [
        {"case": name, "equal": reference[name] == text, "length": len(text)}
        for name, text in ours.items()
    ]
    return {
        "check": "camel_inception_prompts",
        "reference": "ref/camel/camel/prompts/ai_society.py",
        "cases": cases,
        "passed": all(c["equal"] for c in cases),
    }


def check_tot_reference() -> dict:
    """Tier 1 for T9: Game of 24 prompts must match ToT's own source."""
    from kvpim.workloads.t9_tree_search import PROPOSE_PROMPT, VALUE_PROMPT

    reference = load_reference_strings(
        "tree-of-thought-llm/src/tot/prompts/game24.py",
        {"propose_prompt", "value_prompt"},
    )
    pairs = {
        "propose_prompt": PROPOSE_PROMPT,
        "value_prompt": VALUE_PROMPT,
    }
    cases = [
        {"case": name, "equal": reference[name] == text, "length": len(text)}
        for name, text in pairs.items()
    ]
    return {
        "check": "tot_game24_prompts",
        "reference": "ref/tree-of-thought-llm/src/tot/prompts/game24.py",
        "cases": cases,
        "passed": all(c["equal"] for c in cases),
    }


def check_debate_reference() -> dict:
    """Tier 1 for T4: match Du et al.'s own debate prompt assembly.

    The reference keeps the round-0 question prompt inline in `__main__` rather
    than in a constant, so it is checked by looking for our copy among the
    source's string literals; the debate-round message is checked by running the
    reference `construct_message` on synthetic contexts.
    """
    from kvpim.workloads.t4_debate import QUESTION_PROMPT, construct_message

    rel_path = "llm_multiagent_debate/gsm/gen_gsm.py"
    literals = {
        node.value
        for node in ast.walk(ast.parse(reference_source(rel_path)))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    cases = [
        {
            "case": "question_prompt",
            "equal": QUESTION_PROMPT in literals,
            "length": len(QUESTION_PROMPT),
        }
    ]

    reference = load_reference_function(rel_path, "construct_message")
    question = "Janet has 3 apples and eats 1. How many are left?"
    replies = ["Two.\n\n\\boxed{2}", "```code``` \\boxed{2}"]
    # `agents` is a list of per-agent conversations indexed at `idx`.
    idx = 1
    contexts = [[None, {"content": reply}] for reply in replies]
    theirs = reference(contexts, question, idx)
    ours = construct_message(replies, question)
    cases.append(
        {
            "case": "debate_message",
            "equal": theirs["role"] == "user" and theirs["content"] == ours,
            "ours_len": len(ours),
            "theirs_len": len(theirs["content"]),
        }
    )

    return {
        "check": "debate_prompts",
        "reference": f"ref/{rel_path}",
        "cases": cases,
        "passed": all(c["equal"] for c in cases),
    }


def check_autogen_reference() -> dict:
    """Tier 1 for T5: speaker-selection templates must match AutoGen 0.2.34."""
    from kvpim.workloads.t5_groupchat import (
        SELECT_SPEAKER_MESSAGE_TEMPLATE,
        SELECT_SPEAKER_PROMPT_TEMPLATE,
    )

    rel_path = "autogen/autogen/agentchat/groupchat.py"
    rev = "v0.2.34"
    reference = load_reference_strings(
        rel_path,
        {"select_speaker_message_template", "select_speaker_prompt_template"},
        rev=rev,
    )
    pairs = {
        "select_speaker_message_template": SELECT_SPEAKER_MESSAGE_TEMPLATE,
        "select_speaker_prompt_template": SELECT_SPEAKER_PROMPT_TEMPLATE,
    }
    cases = [
        {"case": name, "equal": reference[name] == text, "length": len(text)}
        for name, text in pairs.items()
    ]
    return {
        "check": "autogen_speaker_selection",
        "reference": f"ref/{rel_path}@{rev}",
        "cases": cases,
        "passed": all(c["equal"] for c in cases),
    }


def check_swarm_reference() -> dict:
    """Tier 1 for T6: the one hand-copied agent instruction must match Swarm.

    Four of the five agent prompts are read out of `ref/` at load time and are
    reference-identical by construction; the Flight Modification agent's is
    embedded in a `swarm.Agent(...)` call the driver cannot execute, so it was
    copied and is the only one that can drift.
    """
    from kvpim.workloads.t6_handoff import load_agents

    rel_path = "swarm/examples/airline/configs/agents.py"
    instructions = None
    for node in ast.walk(ast.parse(reference_source(rel_path))):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "flight_modification" not in names:
            continue
        for kw in node.value.keywords:
            if kw.arg == "instructions" and isinstance(kw.value, ast.Constant):
                instructions = kw.value.value
    if instructions is None:
        raise LookupError(f"flight_modification instructions not found in {rel_path}")

    ours = load_agents()["flight_modification"]
    return {
        "check": "swarm_agent_instructions",
        "reference": f"ref/{rel_path}",
        "cases": [
            {
                "case": "flight_modification",
                "equal": instructions == ours,
                "ours_len": len(ours),
                "theirs_len": len(instructions),
            }
        ],
        "passed": instructions == ours,
    }


TIER1_CHECKS = {
    "T1": check_metagpt_reference,
    "T3": check_moa_reference,
    "T4": check_debate_reference,
    "T5": check_autogen_reference,
    "T6": check_swarm_reference,
    "T8": check_camel_reference,
    "T9": check_tot_reference,
}

# The two topologies with no reference implementation to compare against, so
# that an absent Tier 1 result reads as a recorded fact rather than an omission.
TIER1_UNAVAILABLE = {
    "T2": "GAIA supplies tasks, not an orchestrator; the prompts are ours "
    "(plan section 3).",
    "T7": "Reflexion's prompts are entangled with its framework; it exposes no "
    "standalone constants.",
}


_TURN_START = "<|im_start|>"
_TURN_END = "<|im_end|>"


def split_chat_turns(text: str) -> list[tuple[str, str]]:
    """Splits a rendered Qwen chat prompt into ordered (role, content) turns.

    Ordered rather than keyed by role: multi-turn prompts carry several user
    turns and collapsing them would hide exactly the structure being checked.
    """
    turns = []
    for chunk in text.split(_TURN_START)[1:]:
        role, _, body = chunk.partition("\n")
        content = body.split(_TURN_END)[0]
        if role.strip() != "assistant" or content:
            turns.append((role.strip(), content))
    return turns


def turn_content(turns, role: str, index: int = 0) -> str:
    """Returns the content of the `index`-th turn with the given role."""
    matching = [content for name, content in turns if name == role]
    if not matching:
        return ""
    return matching[index if index >= 0 else len(matching) + index]


def check_t3_structure(trace_dir: str | Path, tokenizer=None) -> dict:
    """Tier 2 for T3: structural relations the fan-out topology must satisfy.

    Stated from the topology definition, so a driver that quietly makes prompts
    agent-specific (an index in the user turn, references in a different order)
    fails here even though every measurement downstream would look healthy.

    Args:
        trace_dir: Configuration directory to check.
        tokenizer: Tokenizer used for the run; loaded from the manifest if None.

    Returns:
        One entry per invariant with its violations.
    """
    from kvpim.workloads.t3_fanout import (
        AGGREGATE_AND_SYNTHESIZE,
        PROPOSER_SYSTEM_PROMPTS,
    )

    trace_dir = Path(trace_dir)
    manifest = json.loads((trace_dir / "manifest.json").read_text())
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(manifest["model"])

    calls = read_jsonl(trace_dir / "calls.jsonl")
    decoded = []
    for call in calls:
        turns = split_chat_turns(tokenizer.decode(call["prompt_token_ids"]))
        decoded.append(
            {
                **call,
                "turns": turns,
                "system": turn_content(turns, "system"),
                "user": turn_content(turns, "user", -1),
            }
        )

    violations: dict[str, list] = {
        "persona_prefix": [],
        "layer0_no_injection": [],
        "user_turn_identical": [],
        "injection_identical": [],
        "agg_prompt_verbatim": [],
        "aggregator_no_persona": [],
    }

    def where(call):
        return f"{call['workflow_id']}/call{call['call_idx']}"

    groups: dict[tuple, list] = {}
    for call in decoded:
        if call["agent_id"] == "aggregator":
            continue
        index = call["meta"]["proposer"]
        persona = PROPOSER_SYSTEM_PROMPTS[index]
        if not call["system"].startswith(persona):
            violations["persona_prefix"].append(where(call))
        injected = call["system"][len(persona) :]
        layer = call["meta"]["layer"]
        if layer == 0:
            if injected:
                violations["layer0_no_injection"].append(where(call))
        elif not injected.startswith("\n\n" + AGGREGATE_AND_SYNTHESIZE):
            violations["agg_prompt_verbatim"].append(where(call))
        groups.setdefault((call["workflow_id"], layer), []).append((call, injected))

    for (workflow_id, layer), members in groups.items():
        users = {call["user"] for call, _ in members}
        if len(users) > 1:
            violations["user_turn_identical"].append(f"{workflow_id}/layer{layer}")
        injections = {injected for _, injected in members}
        if layer > 0 and len(injections) > 1:
            violations["injection_identical"].append(f"{workflow_id}/layer{layer}")

    for call in decoded:
        if call["agent_id"] != "aggregator":
            continue
        if not call["system"].startswith(AGGREGATE_AND_SYNTHESIZE):
            violations["agg_prompt_verbatim"].append(where(call))
        if any(persona in call["system"] for persona in PROPOSER_SYSTEM_PROMPTS):
            violations["aggregator_no_persona"].append(where(call))

    return {
        "check": "t3_structure",
        "trace_dir": str(trace_dir),
        "num_calls": len(calls),
        "violations": {k: v for k, v in violations.items() if v},
        "passed": not any(violations.values()),
    }


def decode_calls(trace_dir: str | Path, tokenizer=None) -> tuple[dict, list[dict]]:
    """Loads a run's calls with their prompts decoded back into chat turns."""
    trace_dir = Path(trace_dir)
    manifest = json.loads((trace_dir / "manifest.json").read_text())
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(manifest["model"])
    decoded = []
    for call in read_jsonl(trace_dir / "calls.jsonl"):
        text = tokenizer.decode(call["prompt_token_ids"])
        turns = split_chat_turns(text)
        # A raw-completion call carries no chat markers at all; the decoded text
        # *is* the prompt.
        decoded.append(
            {
                **call,
                "turns": turns,
                "system": turn_content(turns, "system"),
                "user": turn_content(turns, "user", -1) if turns else text,
                "output": tokenizer.decode(call["output_token_ids"]),
                "is_raw": not turns,
            }
        )
    return manifest, decoded


def _by_workflow(decoded: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for call in decoded:
        grouped.setdefault(call["workflow_id"], []).append(call)
    return grouped


def _where(call) -> str:
    return f"{call['workflow_id']}/call{call['call_idx']}"


def _check_t1(decoded: list[dict]) -> dict[str, list]:
    """Waterfall: roles in order, MetaGPT prompts, context only ever grows."""
    from kvpim.workloads.t1_sequential import PIPELINE, role_system_prompt

    bad: dict[str, list] = {"role_system_prompt": [], "pipeline_order": [],
                            "context_monotonic": []}
    for workflow_id, calls in _by_workflow(decoded).items():
        for call in calls:
            role = call["agent_id"]
            if call["system"] != role_system_prompt(role):
                bad["role_system_prompt"].append(_where(call))
        stages = [c["agent_id"] for c in calls[: len(PIPELINE)]]
        if stages != PIPELINE:
            bad["pipeline_order"].append(f"{workflow_id}: {stages}")
        for previous, current in zip(calls[: len(PIPELINE) - 1], calls[1 : len(PIPELINE)]):
            if not current["user"].startswith(previous["user"]):
                bad["context_monotonic"].append(_where(current))
    return bad


def _check_t2(decoded: list[dict]) -> dict[str, list]:
    """Supervisor: workers share one system prompt and the task description."""
    from kvpim.workloads.t2_supervisor import SUPERVISOR_SYSTEM, WORKER_SYSTEM

    bad: dict[str, list] = {"worker_system_identical": [], "supervisor_system": [],
                            "shared_task_prefix": [], "synthesis_has_plan": []}
    for workflow_id, calls in _by_workflow(decoded).items():
        workers = [c for c in calls if c["agent_id"].startswith("worker_")]
        supervisors = [c for c in calls if c["agent_id"] == "supervisor"]
        for call in workers:
            if call["system"] != WORKER_SYSTEM:
                bad["worker_system_identical"].append(_where(call))
        for call in supervisors:
            if call["system"] != SUPERVISOR_SYSTEM:
                bad["supervisor_system"].append(_where(call))
        if workers:
            prefixes = {c["user"].split("\n\nYour subtask:")[0] for c in workers}
            if len(prefixes) > 1:
                bad["shared_task_prefix"].append(f"{workflow_id}: {len(prefixes)} variants")
        if len(supervisors) > 1:
            synthesis = supervisors[-1]
            if not any(role == "assistant" for role, _ in synthesis["turns"]):
                bad["synthesis_has_plan"].append(_where(synthesis))
    return bad


def _check_t4(decoded: list[dict]) -> dict[str, list]:
    """Debate: agents answer the same question alone, then read only the others.

    The peer block is checked through the reference's own ```-fenced wrapper
    rather than by substring, so a short reply that happens to occur inside
    another cannot read as either a present peer or a leaked self.
    """
    from kvpim.workloads.t4_debate import NUM_AGENTS, NUM_ROUNDS

    bad: dict[str, list] = {"call_count": [], "round0_prompt_identical": [],
                            "context_monotonic": [], "peer_present": [],
                            "self_excluded": [], "peer_order": [],
                            "note_replies_identical": []}
    for workflow_id, calls in _by_workflow(decoded).items():
        if len(calls) != NUM_AGENTS * NUM_ROUNDS:
            bad["call_count"].append(f"{workflow_id}: {len(calls)}")
            continue
        rounds = [calls[r * NUM_AGENTS : (r + 1) * NUM_AGENTS] for r in range(NUM_ROUNDS)]
        # Round 0 is the independent pass: one question, no peer information.
        if len({c["user"] for c in rounds[0]}) > 1:
            bad["round0_prompt_identical"].append(workflow_id)
        for index, (previous_round, current_round) in enumerate(zip(rounds, rounds[1:])):
            # An agent's reply is read back off its own recorded conversation
            # rather than from `output`, which still carries the end-of-turn
            # marker that the orchestrator strips before quoting it.
            replies = [
                [c for role, c in call["turns"] if role == "assistant"][index]
                for call in current_round
            ]
            if len(set(replies)) < len(replies):
                bad["note_replies_identical"].append(f"{workflow_id} round{index}")
            for agent, call in enumerate(current_round):
                if not call["turns"][0][1].startswith(previous_round[agent]["turns"][0][1]):
                    bad["context_monotonic"].append(_where(call))
                positions = []
                for peer, reply in enumerate(replies):
                    fenced = f"```{reply}```"
                    if peer == agent:
                        # Undecidable when a peer said exactly the same thing.
                        if fenced in call["user"] and replies.count(reply) == 1:
                            bad["self_excluded"].append(_where(call))
                        continue
                    at = call["user"].find(fenced)
                    if at < 0:
                        bad["peer_present"].append(f"{_where(call)}: agent {peer}")
                    else:
                        positions.append(at)
                if positions != sorted(positions):
                    bad["peer_order"].append(_where(call))
    return bad


def _check_t5(decoded: list[dict]) -> dict[str, list]:
    """Group chat: one broadcast transcript, a different system prompt in front.

    The manager sees the same transcript as the speaker plus its own trailing
    selection prompt — that identity is the topology, and it is what makes every
    speaker's prefix fork at token zero.
    """
    from kvpim.workloads.t5_groupchat import (
        AGENTS,
        MAX_ROUND,
        SELECT_SPEAKER_MESSAGE_TEMPLATE,
        SELECT_SPEAKER_PROMPT_TEMPLATE,
    )

    agentlist = "[" + ", ".join(AGENTS) + "]"
    selection_prompt = SELECT_SPEAKER_PROMPT_TEMPLATE.format(agentlist=agentlist)
    head = SELECT_SPEAKER_MESSAGE_TEMPLATE.split("{roles}")[0]
    bad: dict[str, list] = {"stage_alternates": [], "manager_system": [],
                            "speaker_known": [], "speaker_system": [],
                            "shared_transcript": [], "transcript_monotonic": [],
                            "round_cap": []}
    for workflow_id, calls in _by_workflow(decoded).items():
        if len(calls) > 2 * MAX_ROUND:
            bad["round_cap"].append(f"{workflow_id}: {len(calls)}")
        for index, call in enumerate(calls):
            expected = "select_speaker" if index % 2 == 0 else "speak"
            if call["meta"]["stage"] != expected:
                bad["stage_alternates"].append(_where(call))
        for manager, speaker in zip(calls[::2], calls[1::2]):
            if not (
                manager["system"].startswith(head)
                and agentlist in manager["system"]
            ):
                bad["manager_system"].append(_where(manager))
            if speaker["agent_id"] not in AGENTS:
                bad["speaker_known"].append(_where(speaker))
            elif speaker["system"] != AGENTS[speaker["agent_id"]]:
                bad["speaker_system"].append(_where(speaker))
            manager_body = [c for role, c in manager["turns"] if role != "system"]
            speaker_body = [c for role, c in speaker["turns"] if role != "system"]
            if manager_body[-1:] != [selection_prompt]:
                bad["manager_system"].append(_where(manager))
            elif manager_body[:-1] != speaker_body:
                bad["shared_transcript"].append(_where(speaker))
        for previous, current in zip(calls[1::2], calls[3::2]):
            previous_body = [c for role, c in previous["turns"] if role != "system"]
            current_body = [c for role, c in current["turns"] if role != "system"]
            if current_body[: len(previous_body)] != previous_body:
                bad["transcript_monotonic"].append(_where(current))
    return bad


def _check_t6(decoded: list[dict]) -> dict[str, list]:
    """Handoff: only the system prompt is swapped; the history is never trimmed.

    The no-trim property is asserted specifically across the turns where the
    agent changes, since that is the one place a framework would normally reset.
    """
    from kvpim.workloads.t6_handoff import HANDOFFS, MAX_TURNS, load_agents

    agents = load_agents()
    bad: dict[str, list] = {"stage_alternates": [], "agent_system": [],
                            "legal_handoff": [], "history_monotonic": [],
                            "no_trim_on_handoff": [], "turn_cap": []}
    for workflow_id, calls in _by_workflow(decoded).items():
        if len(calls) > MAX_TURNS:
            bad["turn_cap"].append(f"{workflow_id}: {len(calls)}")
        for index, call in enumerate(calls):
            expected = "user" if index % 2 == 0 else "agent"
            if call["meta"]["stage"] != expected:
                bad["stage_alternates"].append(_where(call))
        agent_calls = [c for c in calls if c["meta"]["stage"] == "agent"]
        for call in agent_calls:
            if call["system"] != agents.get(call["agent_id"]):
                bad["agent_system"].append(_where(call))
        for previous, current in zip(agent_calls, agent_calls[1:]):
            source, target = previous["agent_id"], current["agent_id"]
            if target != source and target not in HANDOFFS.get(source, set()):
                bad["legal_handoff"].append(f"{_where(current)}: {source}->{target}")
            previous_body = [c for role, c in previous["turns"] if role != "system"]
            current_body = [c for role, c in current["turns"] if role != "system"]
            if current_body[: len(previous_body)] != previous_body:
                key = "no_trim_on_handoff" if target != source else "history_monotonic"
                bad[key].append(_where(current))
    return bad


def _check_t7(decoded: list[dict]) -> dict[str, list]:
    """Reflexion: actor/evaluator alternate and memory keeps one reflection."""
    bad: dict[str, list] = {"stage_cycle": [], "memory_omega": []}
    for workflow_id, calls in _by_workflow(decoded).items():
        stages = [c["meta"]["stage"] for c in calls]
        for previous, current in zip(stages, stages[1:]):
            legal = {
                "actor": {"evaluator"},
                "evaluator": {"reflection", "actor"},
                "reflection": {"actor"},
            }
            if current not in legal.get(previous, set()):
                bad["stage_cycle"].append(f"{workflow_id}: {previous}->{current}")
        for call in calls:
            if call["meta"]["stage"] == "actor" and call["user"].count("## reflection") > 1:
                bad["memory_omega"].append(_where(call))
    return bad


def _check_t8(decoded: list[dict]) -> dict[str, list]:
    """Role-play: mirrored transcripts, own rule block, bounded message count."""
    bad: dict[str, list] = {"rules_block": [], "shared_task": [], "message_cap": [],
                            "mirrored_transcript": []}
    from kvpim.workloads.t8_roleplay import MAX_MESSAGES

    for workflow_id, calls in _by_workflow(decoded).items():
        if len(calls) > MAX_MESSAGES:
            bad["message_cap"].append(f"{workflow_id}: {len(calls)}")
        tasks = set()
        for call in calls:
            header = (
                "===== RULES OF USER ====="
                if call["agent_id"] == "ai_user"
                else "===== RULES OF ASSISTANT ====="
            )
            if not call["system"].startswith(header):
                bad["rules_block"].append(_where(call))
            marker = "Here is the task: "
            if marker in call["system"]:
                tasks.add(call["system"].split(marker, 1)[1].split(". Never forget")[0])
        if len(tasks) > 1:
            bad["shared_task"].append(f"{workflow_id}: {len(tasks)} task strings")
        for user_call, assistant_call in zip(calls, calls[1:]):
            if user_call["agent_id"] != "ai_user" or assistant_call["agent_id"] != "ai_assistant":
                continue
            user_body = [c for r, c in user_call["turns"] if r != "system"]
            assistant_body = [c for r, c in assistant_call["turns"] if r != "system"]
            if assistant_body[: len(user_body)] != user_body:
                bad["mirrored_transcript"].append(_where(assistant_call))
    return bad


def _check_t9(decoded: list[dict]) -> dict[str, list]:
    """Tree search: every prompt is the paper's template with only input filled."""
    from kvpim.workloads.t9_tree_search import (
        PROPOSE_PROMPT,
        VALUE_PROMPT,
        VALUE_WEIGHTS,
        parse_candidates,
    )

    templates = {
        "expander": PROPOSE_PROMPT.split("{input}"),
        "evaluator": VALUE_PROMPT.split("{input}"),
    }
    bad: dict[str, list] = {"prompt_template": [], "raw_completion": [],
                            "output_parseable": [], "note_value_no_verdict": []}
    for call in decoded:
        # ToT's prompts are few-shot completions; the chat template must not wrap
        # them, or an instruct model answers the question instead of continuing
        # the pattern and the parser reads nothing.
        if not call["is_raw"]:
            bad["raw_completion"].append(_where(call))
        head, tail = templates[call["agent_id"]]
        # Trailing newlines are compared stripped on both sides: the propose
        # template ends in one and the value template's tail *is* one, so a
        # decoder that drops it must not read as a template mismatch.
        prompt, tail = call["user"].rstrip("\n"), tail.rstrip("\n")
        if not (prompt.startswith(head) and prompt.endswith(tail)):
            bad["prompt_template"].append(_where(call))
        # A prompt in the right shape is not enough: an unreadable propose reply
        # kills the branch, and the search silently stops at its first node.
        # An unreadable *value* reply does not — ToT's own value_outputs_unwrap
        # scores a reply with no verdict as 0 and carries on, so those are
        # counted rather than treated as a violation.
        if call["agent_id"] == "expander":
            if not parse_candidates(call["output"]):
                bad["output_parseable"].append(_where(call))
        elif not any(
            line.strip().lower() in VALUE_WEIGHTS
            for line in call["output"].splitlines()
        ):
            bad["note_value_no_verdict"].append(_where(call))
    return bad


TIER2_CHECKS = {
    "T1": _check_t1,
    "T2": _check_t2,
    "T4": _check_t4,
    "T5": _check_t5,
    "T6": _check_t6,
    "T7": _check_t7,
    "T8": _check_t8,
    "T9": _check_t9,
}


def check_structure(trace_dir: str | Path, tokenizer=None) -> dict:
    """Tier 2 dispatcher: runs the invariants of whichever topology this is."""
    manifest, decoded = decode_calls(trace_dir, tokenizer)
    topology = manifest["topology"]
    if topology == "T3":
        return check_t3_structure(trace_dir, tokenizer)
    checker = TIER2_CHECKS.get(topology)
    if checker is None:
        return {"check": "structure", "topology": topology, "passed": None,
                "note": "no invariants defined"}
    found = checker(decoded)
    # A `note_` key records something worth knowing that the reference
    # implementation itself tolerates, so it is reported without failing.
    violations = {k: v for k, v in found.items() if v and not k.startswith("note_")}
    notes = {k[5:]: v for k, v in found.items() if v and k.startswith("note_")}
    return {
        "check": f"{topology.lower()}_structure",
        "trace_dir": str(trace_dir),
        "num_calls": len(decoded),
        "violations": violations,
        "notes": {k: len(v) for k, v in notes.items()},
        "passed": not violations,
    }


# ---------------------------------------------------------------------------
# The measurement itself (plan section 4): how many stored blocks are the same
# as some other stored block.
#
# Candidate pairs are formed only between blocks whose token content is
# *identical* — a block with different tokens is not a deduplication candidate
# at all, and comparing them would import exactly the failure mode ContextPilot
# warns about. Within a candidate group the decision is numeric: cosine at or
# above the layer's own tau, which sanity #6 calibrated from recomputation noise.
# ---------------------------------------------------------------------------

# Per-layer tau from the sanity #6 calibration (plan section 4). Recalibrate on
# a real trace before the headline numbers are taken.
TAU_BY_LAYER = {0: 0.99999, 8: 0.99998, 17: 0.97809, 26: 0.96400, 35: 0.99648}
VIEWS = ("K_post", "K_derope", "V")


def index_dumped_blocks(trace_dir: str | Path) -> list[dict]:
    """Lists every dumped block with where its tensors live.

    Returns:
        One record per block: workflow, call index, row inside that call's
        tensors, block hash and the positions its tokens occupied.
    """
    dumps = Path(trace_dir) / "dumps"
    records = []
    for meta_path in sorted(dumps.glob("*/*_meta.json")):
        meta = json.loads(meta_path.read_text())
        for row, block in enumerate(meta["blocks"]):
            records.append(
                {
                    "workflow_id": meta["workflow_id"],
                    "call_idx": meta["call_idx"],
                    "row": row,
                    "block_hash": block["block_hash"],
                    "positions": block["positions"],
                    "dir": meta_path.parent,
                }
            )
    return records


def _load_rows(path: Path, key: str, rows: list[int]):
    """Reads specific block rows out of a safetensors file.

    One open per call, not one per row: at the finest slicing a group can hold
    thousands of units and the per-open cost dominates everything else.
    """
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        tensor = handle.get_slice(key)
        return {row: tensor[row : row + 1][0] for row in rows}


def _cosine_matrix(vectors) -> "object":
    import torch

    flat = torch.stack([v.reshape(-1).double() for v in vectors])
    flat = flat / flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return flat @ flat.T


def count_duplicates(
    trace_dir: str | Path,
    tau: dict[int, float] | None = None,
    thresholds: tuple[float, ...] = (0.99, 0.995, 0.999, 0.9999),
    slice_to: int | None = None,
) -> dict:
    """Counts how many stored blocks duplicate another one.

    Args:
        trace_dir: An ample-tier configuration directory holding `dumps/`.
        tau: Per-layer decision threshold; defaults to the sanity #6 calibration.
        thresholds: Extra fixed thresholds reported for the sensitivity appendix.
        slice_to: Re-cut each dumped block into units of this many tokens before
            counting. The 4 and 8 tiers run 16-token physical blocks and differ
            only in matching granularity, so they carry no dump of their own;
            the plan (section 6.1) answers them by re-cutting the 16 tier's
            tensors. Same data, finer cut, which isolates granularity as the
            only variable — note this is the 16 tier's token stream, which for
            topologies whose control flow depends on generated text is not
            byte-identical to what the 4 tier itself produced.

    Returns:
        Per-view, per-layer duplicate counts plus the strict cross-layer count
        and the deduplication saving.
    """
    import torch

    from kvpim.derope import RopeParams, derope

    trace_dir = Path(trace_dir)
    manifest = json.loads((trace_dir / "manifest.json").read_text())
    layers = manifest["sample_layers"]
    tau = tau or TAU_BY_LAYER
    params = RopeParams()

    content = {h: tuple(b["token_ids"]) for h, b in
               working_set_blocks(read_jsonl(trace_dir / "blocks.jsonl")).items()}
    blocks = index_dumped_blocks(trace_dir)

    # A "unit" is what gets counted: a whole dumped block, or one slice of it.
    # `offset` says where inside the block's tensor rows the unit starts.
    units = []
    for block in blocks:
        tokens = content.get(block["block_hash"])
        if not tokens:
            continue
        span = min(len(tokens), len(block["positions"]))
        step = slice_to or span
        for start in range(0, span, step):
            stop = min(start + step, span)
            units.append(
                {
                    **block,
                    "offset": start,
                    "length": stop - start,
                    "tokens": tokens[start:stop],
                    "positions": block["positions"][start:stop],
                }
            )
    n_total = len(units)

    # Only identical token content can be a duplicate, so the comparison set is
    # the groups with more than one member; singletons cost nothing.
    groups: dict[tuple, list[int]] = {}
    for i, unit in enumerate(units):
        groups.setdefault(unit["tokens"], []).append(i)
    candidate_groups = [g for g in groups.values() if len(g) > 1]

    dup = {view: {layer: set() for layer in layers} for view in VIEWS}
    dup_all_layers = {view: set() for view in VIEWS}
    dup_any_layer = {view: set() for view in VIEWS}
    # Fixed thresholds for the appendix, kept per layer so they are directly
    # comparable with the per-layer tau counts.
    sweep = {view: {t: {layer: set() for layer in layers} for t in thresholds}
             for view in VIEWS}
    pairs_examined = 0

    for members in candidate_groups:
        per_layer_hits = {view: [] for view in VIEWS}
        for layer in layers:
            vectors = {view: [] for view in VIEWS}
            by_file: dict[Path, list[int]] = {}
            for index in members:
                unit = units[index]
                path = unit["dir"] / f"{unit['call_idx']:04d}_{layer:02d}.safetensors"
                by_file.setdefault(path, []).append(index)
            loaded = {}
            for path, indices in by_file.items():
                rows = sorted({units[i]["row"] for i in indices})
                loaded[path] = (
                    _load_rows(path, "k", rows),
                    _load_rows(path, "v", rows),
                )
            for index in members:
                unit = units[index]
                path = unit["dir"] / f"{unit['call_idx']:04d}_{layer:02d}.safetensors"
                k_rows, v_rows = loaded[path]
                k = k_rows[unit["row"]].float()
                v = v_rows[unit["row"]].float()
                lo, hi = unit["offset"], unit["offset"] + unit["length"]
                positions = torch.tensor(unit["positions"])
                vectors["K_post"].append(k[lo:hi])
                vectors["V"].append(v[lo:hi])
                vectors["K_derope"].append(derope(k[lo:hi], positions, params))

            for view in VIEWS:
                similarity = _cosine_matrix(vectors[view])
                similarity.fill_diagonal_(-1.0)
                best = similarity.max(dim=1).values
                hit = best >= tau[layer]
                per_layer_hits[view].append(hit)
                for position, flag in enumerate(hit.tolist()):
                    if flag:
                        dup[view][layer].add(members[position])
                        dup_any_layer[view].add(members[position])
                for threshold in thresholds:
                    for position, flag in enumerate((best >= threshold).tolist()):
                        if flag:
                            sweep[view][threshold][layer].add(members[position])
            pairs_examined += len(members) * (len(members) - 1) // 2

        for view in VIEWS:
            stacked = torch.stack(per_layer_hits[view])
            for position, flag in enumerate(stacked.all(dim=0).tolist()):
                if flag:
                    dup_all_layers[view].add(members[position])

    def saving(marked: set) -> float:
        return round(len(marked) / n_total, 6) if n_total else 0.0

    return {
        "trace_dir": str(trace_dir),
        "topology": manifest["topology"],
        "block_tier": slice_to or manifest["block_tier"],
        "sliced_from": manifest["block_tier"] if slice_to else None,
        "capacity_tier": manifest["capacity_tier"],
        "n_total": n_total,
        "n_candidate_groups": len(candidate_groups),
        "n_blocks_in_groups": sum(len(g) for g in candidate_groups),
        "pairs_examined": pairs_examined,
        "tau": {str(k): v for k, v in tau.items()},
        "n_dup_per_layer": {
            view: {str(layer): len(ids) for layer, ids in per_layer.items()}
            for view, per_layer in dup.items()
        },
        "n_dup_strict_all_layers": {v: len(ids) for v, ids in dup_all_layers.items()},
        "n_dup_any_layer": {v: len(ids) for v, ids in dup_any_layer.items()},
        "dup_fraction_strict": {v: saving(ids) for v, ids in dup_all_layers.items()},
        "dup_fraction_any_layer": {v: saving(ids) for v, ids in dup_any_layer.items()},
        "sensitivity_per_layer": {
            view: {
                str(t): {str(layer): len(ids) for layer, ids in per_layer.items()}
                for t, per_layer in per_threshold.items()
            }
            for view, per_threshold in sweep.items()
        },
    }


def build_counts_table(
    traces_root: str | Path,
    out_path: str | Path | None = None,
    tau: dict[int, float] | None = None,
) -> "object":
    """Runs the count over every ample configuration and tabulates it.

    Args:
        traces_root: Directory holding the per-configuration trace directories.
        out_path: Where to write `counts.parquet`; skipped when None.
        tau: Per-layer threshold override.

    Returns:
        A dataframe with one row per configuration and view.
    """
    import pandas as pd

    rows = []
    for trace_dir in sorted(Path(traces_root).glob("*/")):
        if not (trace_dir / "dumps").is_dir() or not (trace_dir / "manifest.json").exists():
            continue
        result = count_duplicates(trace_dir, tau=tau)
        for view in VIEWS:
            row = {
                "topology": result["topology"],
                "block_tier": result["block_tier"],
                "capacity_tier": result["capacity_tier"],
                "view": view,
                "n_total": result["n_total"],
                "n_candidate_groups": result["n_candidate_groups"],
                "n_blocks_in_groups": result["n_blocks_in_groups"],
                "n_dup_strict": result["n_dup_strict_all_layers"][view],
                "n_dup_any_layer": result["n_dup_any_layer"][view],
                "dup_fraction_strict": result["dup_fraction_strict"][view],
                "dup_fraction_any_layer": result["dup_fraction_any_layer"][view],
            }
            for layer, count in result["n_dup_per_layer"][view].items():
                row[f"n_dup_layer_{layer}"] = count
            rows.append(row)

    frame = pd.DataFrame(rows)
    if out_path is not None and not frame.empty:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out_path, index=False)
    return frame


_CONFIG_README = """# {topology} · block tier {block_tier} · {capacity_tier}

自动生成于 {timestamp}，由 `kvpim.analyze.write_config_readme` 从本目录的
`manifest.json` / `sanity.log` / `blocks.jsonl` 读出。**不要手改**，重跑会覆盖。

## 这一组跑了什么

| 项 | 值 |
|---|---|
| 拓扑 | {topology} — {topology_desc} |
| benchmark | {benchmark} |
| block 档 | {block_tier} token（物理 `block_size={block_size}`，`prefix_match_unit={prefix_match_unit}`）|
| 容量档 | {capacity_tier}{capacity_detail} |
| 模型 | {model} |
| 采样 | `temperature=0`、`top_p=1`、`top_k=0`、`max_tokens={max_tokens}`、`generation_config="vllm"` |
| workflow / 调用 | {num_workflows} / {num_calls} |
| KV dump | {dump_state} |
| vLLM commit | `{vllm_commit}` |
| 耗时 | {elapsed_min} 分钟 |

## 怎么复现

```bash
# 环境见 notes/planfiles/athena-execution-plan.md §3（必须源码编译，不能用预编译轮子）
cd /home/cw636/chenyi/KVPIM/scratch
sbatch -p athena-small -w node4 --job-name={topology}-b{block_tier}-{capacity_tier} \\
  --export=ALL,TOPOLOGY={topology},BLOCK_TIER={block_tier},CAPACITY={capacity_tier},\\
NUM_TASKS={num_workflows},ZMQ_PORT=<未占用端口> run_config.sbatch
```

受限档会在启动时读取同拓扑同档充足档的 `manifest.json` 自行计算容量，
所以**必须先跑充足档**。抽题由 `seed=0` 固定，同一档的两个容量档用同一批题。

## 产出

| 文件 | 内容 |
|---|---|
| `manifest.json` | 全部配置参数、实测工作集、充足档判据结果 |
| `calls.jsonl` | 每次调用：agent、DAG 边、完整 `prompt_token_ids`、输出、命中 token 数、起止时刻 |
| `blocks.jsonl` | 全部 `BlockStored` / `BlockRemoved` / `AllBlocksCleared` 事件 |
| `sanity.log` | sanity #4（事件流对账）与 #5（拓扑正确性 Tier 1/Tier 2）的结果 |
| `dumps/` | {dump_state}。`<workflow>/<call>_<layer>.safetensors` 存 K 与 V，同名 `_meta.json` 存每块的 `(block_id, block_hash, positions)` |

## 实测结果

### 工作集与容量

```
每个 workflow 的工作集（token）: {per_workflow}
  平均 {w_mean}   最大 {w_max}
驱逐掉的 block 数: {num_removed}
充足档判据（单 workflow 内不驱逐）: {ample_ok}
```

{capacity_note}

### Sanity

{sanity_summary}

### 重复度计数

{counts_summary}

## 口径提醒

- `n_blocks_in_groups` 是**天花板**：只有 token 内容完全相同的块才是去重候选。
- `n_dup_*` 是在候选内用 cosine ≥ 逐层 τ 判出来的。τ 由 sanity #6 从重算噪声标定，
  **当前是工作值**，正式重标前不要把绝对数字当结论。
- `K_post` 是缓存里实际存的；`K_derope` 是离线逆旋转后的；`V` 不过 RoPE，是位置无关的对照。
"""

_TOPOLOGY_DESC = {
    "T1": "Sequential / Pipeline（MetaGPT 五级瀑布）",
    "T2": "Supervisor / Orchestrator-Worker",
    "T3": "Fan-out / Mixture-of-Agents",
    "T4": "Debate（Du et al.）",
    "T5": "Group Chat（AutoGen 动态 GroupChat）",
    "T6": "Handoff / Swarm",
    "T7": "Reflection / Generator-Critic（Reflexion）",
    "T8": "Role-play / Simulation（CAMEL）",
    "T9": "Tree Search（Tree of Thoughts）",
}
_BENCHMARK = {
    "T1": "HumanEval（按 canonical solution 行数分层抽 10 题）",
    "T2": "GAIA validation 无附件题，L1:4 / L2:4 / L3:2",
    "T3": "AlpacaEval 2.0（805 条按指令长度分层抽 10）",
    "T4": "GSM8K test（seed=0 打乱后取前 10）",
    "T5": "MATH level-5（六个非几何科目）",
    "T6": "τ-bench airline 50 个任务等距抽 10",
    "T7": "HumanEval，优先取 baseline 首轮失败的题",
    "T8": "CAMEL AI Society role pair（按 assistant 角色分层）",
    "T9": "Game of 24，ToT 原文 901–1000 切片等距抽 10",
}


def write_config_readme(trace_dir: str | Path, counts: dict | None = None) -> Path:
    """Writes a README describing one configuration, its provenance and results.

    Args:
        trace_dir: The configuration directory to document.
        counts: A `count_duplicates` result to embed; computed when omitted and
            the directory has dumps.

    Returns:
        The path written.
    """
    import time

    trace_dir = Path(trace_dir)
    manifest = json.loads((trace_dir / "manifest.json").read_text())
    working = manifest["working_set"]
    extra = manifest.get("extra") or {}

    sanity_path = trace_dir / "sanity.log"
    if sanity_path.exists():
        sanity = json.loads(sanity_path.read_text())
        lines = []
        s4 = sanity.get("sanity4") or {}
        if s4:
            # `within_one_block` is the pre-two-mode name for `within_tolerance`.
            tolerated = s4.get("within_tolerance", s4.get("within_one_block"))
            mode = f"（{s4['mode']}）" if s4.get("mode") else ""
            lines.append(
                f"- **#4 事件流对账**{mode}：{'PASS' if s4['passed'] else 'FAIL'} — "
                f"{s4['exact_matches']}/{s4['num_calls']} 次调用精确一致，"
                f"{tolerated} 次在容差内"
            )
        for key, label in (("sanity5_tier1", "#5 Tier 1（对论文原文逐字节）"),
                           ("sanity5_tier2", "#5 Tier 2（拓扑结构不变量）")):
            block = sanity.get(key)
            if not block:
                lines.append(f"- **{label}**：未记录")
            elif block.get("reason"):
                lines.append(f"- **{label}**：不适用 — {block['reason']}")
            elif block.get("passed") is None:
                lines.append(f"- **{label}**：未定义不变量")
            else:
                if "cases" in block:
                    detail = f"{len(block['cases'])} 个用例全部逐字节相等"
                else:
                    detail = f"违规 {block.get('violations') or '无'}"
                    if block.get("notes"):
                        detail += f"；已记录但不判失败 {block['notes']}"
                lines.append(
                    f"- **{label}**：{'PASS' if block['passed'] else 'FAIL'} — {detail}"
                )
        sanity_summary = "\n".join(lines)
    else:
        sanity_summary = "_本组尚未产出 `sanity.log`。_"

    if counts is None and (trace_dir / "dumps").is_dir():
        try:
            counts = count_duplicates(trace_dir)
        except Exception as error:  # noqa: BLE001 - a README must still be written
            counts = {"error": f"{type(error).__name__}: {error}"}
    if counts and "error" not in counts:
        header = "| 视角 | " + " | ".join(
            f"L{layer}" for layer in manifest["sample_layers"]
        ) + " | 全层严格 | 任一层 |"
        rule = "|---" * (len(manifest["sample_layers"]) + 3) + "|"
        body = []
        for view in VIEWS:
            per_layer = counts["n_dup_per_layer"][view]
            cells = " | ".join(
                str(per_layer[str(layer)]) for layer in manifest["sample_layers"]
            )
            body.append(
                f"| `{view}` | {cells} | {counts['n_dup_strict_all_layers'][view]} "
                f"| {counts['n_dup_any_layer'][view]} |"
            )
        counts_summary = (
            f"```\nN_total（存过的唯一 block）      = {counts['n_total']}\n"
            f"内容相同的候选组               = {counts['n_candidate_groups']}\n"
            f"落在候选组里的 block（天花板） = {counts['n_blocks_in_groups']}"
            f"  （占 N_total 的 "
            f"{counts['n_blocks_in_groups'] / max(counts['n_total'], 1):.1%}）\n"
            f"实际比对的 block 对             = {counts['pairs_examined']}\n```\n\n"
            + "\n".join([header, rule, *body])
        )
    elif counts:
        counts_summary = f"_计数失败：{counts['error']}_"
    else:
        counts_summary = "_本档不 dump KV 张量（4/8 档复用 16 档，见计划 §6.3），无计数。_"

    capacity_detail = ""
    capacity_note = ""
    if manifest["capacity_tier"] == "limited":
        capacity_detail = (
            f"，`num_gpu_blocks_override={manifest['num_gpu_blocks_override']}`"
            f" = {manifest['max_model_len']} token"
        )
        if extra.get("capacity_floor_applied"):
            capacity_note = (
                f"⚠️ **KVFlow 的 0.5 比例未能达成**：该拓扑最长单次调用 "
                f"{extra['longest_ample_call_tokens']} token，比 0.5×mean(W) 还大，"
                f"池子被下限抬到实际比例 **{extra['achieved_ratio_vs_mean']}**。"
                f"不抬的话超长请求会被引擎直接拒绝。"
            )
        else:
            capacity_note = (
                f"容量按 `0.5 × mean(W)` 定，实际比例 "
                f"{extra.get('achieved_ratio_vs_mean', '—')}。"
            )
    elif not manifest["ample_criterion_met"]:
        capacity_note = (
            "⚠️ **充足档判据未满足**：本组发生了驱逐，说明有 workflow 撑破了 KV 池，"
            "该组的事件层面指标（命中率、W）受污染。"
        )

    text = _CONFIG_README.format(
        topology=manifest["topology"],
        topology_desc=_TOPOLOGY_DESC.get(manifest["topology"], "—"),
        benchmark=_BENCHMARK.get(manifest["topology"], "—"),
        block_tier=manifest["block_tier"],
        block_size=manifest["block_size"],
        prefix_match_unit=manifest["prefix_match_unit"],
        capacity_tier=manifest["capacity_tier"],
        capacity_detail=capacity_detail,
        capacity_note=capacity_note,
        model=manifest["model"],
        max_tokens=manifest["max_tokens"],
        num_workflows=manifest["num_workflows"],
        num_calls=manifest["num_calls"],
        dump_state=(
            f"已 dump {manifest['num_blocks_dumped']} 个 block × "
            f"{len(manifest['sample_layers'])} 层"
            if manifest["dump_enabled"]
            else "本档不 dump"
        ),
        vllm_commit=manifest["vllm_commit"],
        elapsed_min=round(manifest["elapsed_s"] / 60, 1),
        per_workflow=working["per_workflow"],
        w_mean=working["mean"],
        w_max=working["max"],
        num_removed=working["num_blocks_removed"],
        ample_ok=manifest["ample_criterion_met"],
        sanity_summary=sanity_summary,
        counts_summary=counts_summary,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    path = trace_dir / "README.md"
    path.write_text(text)
    return path


def working_set_blocks(events: list[dict]) -> dict[object, dict]:
    """Distinct blocks the run stored, keyed by block hash."""
    blocks: dict[object, dict] = {}
    for event in events:
        if event["type"] != "BlockStored":
            continue
        size = event["block_size"]
        token_ids = event["token_ids"]
        parent = event["parent_block_hash"]
        for i, block_hash in enumerate(event["block_hashes"]):
            blocks.setdefault(
                block_hash,
                {
                    "token_ids": token_ids[i * size : (i + 1) * size],
                    "parent": parent,
                    "block_size": size,
                    "ts": event["ts"],
                },
            )
            parent = block_hash
    return blocks


def evicted_blocks(events: list[dict]) -> set:
    """Block hashes that a `BlockRemoved` event reported."""
    removed = set()
    for event in events:
        if event["type"] == "BlockRemoved":
            removed.update(event["block_hashes"])
    return removed
