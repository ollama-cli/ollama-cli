"""
Planner — decides whether to handle a task directly or delegate to subagents.

The orchestrator calls plan() before processing a user prompt. If the task
can benefit from parallel subagents, it returns a list of subtasks.
Otherwise it returns None and the orchestrator handles it directly.

Subtasks may have dependencies: subtask B can depend on subtask A's result.
The orchestrator executes them in waves — independent tasks first, then
dependent tasks with prior results injected as context.
"""

import json
import re
import logging
from typing import List, Optional

from ollama_cli.core.ollama import OllamaClient

logger = logging.getLogger("ollama-cli.planner")

PLANNING_PROMPT = """Analyze the user's request and decide how to handle it.

If the task is SIMPLE (a single question, greeting, or direct request), respond with exactly:
DIRECT

If the task is COMPLEX, respond with a JSON array of subtasks. Each subtask has:
- "task": what the subagent should do (a clear, self-contained instruction)
- "tools": list of tool names needed (from: web_search, read_file, write_file, list_directory, grep_search, run_python, code_analyze_file)
- "depends_on": (optional) list of subtask indices (0-based) that must finish first

Example for "compare weather in Cork and Dublin":
[
  {"task": "Search for the current weather in Cork, Ireland", "tools": ["web_search"]},
  {"task": "Search for the current weather in Dublin, Ireland", "tools": ["web_search"]}
]

Example for "find the population of France's capital city":
[
  {"task": "Find the capital city of France", "tools": ["web_search"]},
  {"task": "Find the current population of the capital city found in subtask 0", "tools": ["web_search"], "depends_on": [0]}
]

Example for "what is 2+2":
DIRECT

RULES:
- Only delegate if there are 2+ subtasks
- Subtasks WITHOUT depends_on run in parallel
- Subtasks WITH depends_on wait for those subtasks to finish first
- Do NOT delegate simple questions, greetings, or single-step tasks
- Maximum 4 subtasks
- Respond with ONLY "DIRECT" or a JSON array, nothing else"""


def plan(client: OllamaClient, model: str, user_input: str) -> Optional[List[dict]]:
    """Decide whether to delegate a task to subagents.

    Returns a list of subtask dicts, or None for direct handling.
    Each subtask: {"task": str, "tools": [...], "depends_on": [int, ...]}
    """
    messages = [
        {"role": "system", "content": PLANNING_PROMPT},
        {"role": "user", "content": user_input},
    ]

    try:
        response = ""
        for chunk in client.chat(messages, model):
            response += chunk
    except Exception as e:
        logger.warning(f"Planning failed: {e}")
        return None

    response = response.strip()

    if response.upper().startswith("DIRECT"):
        return None

    try:
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            subtasks = json.loads(json_match.group())
            if isinstance(subtasks, list) and len(subtasks) > 0:
                valid = []
                for i, st in enumerate(subtasks[:4]):
                    if isinstance(st, dict) and "task" in st:
                        deps = st.get("depends_on", [])
                        # Validate deps are valid indices
                        if isinstance(deps, list):
                            deps = [d for d in deps if isinstance(d, int) and 0 <= d < i]
                        else:
                            deps = []
                        valid.append({
                            "task": str(st["task"]),
                            "tools": st.get("tools", ["web_search"]),
                            "depends_on": deps,
                        })
                if len(valid) >= 2:
                    return valid
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def get_execution_waves(subtasks: List[dict]) -> List[List[int]]:
    """Group subtasks into execution waves based on dependencies.

    Returns list of waves, each wave is a list of subtask indices
    that can run in parallel.

    Example: [{0, 1}, {2}] means subtasks 0,1 run first, then 2.
    """
    n = len(subtasks)
    completed = set()
    waves = []

    while len(completed) < n:
        wave = []
        for i in range(n):
            if i in completed:
                continue
            deps = set(subtasks[i].get("depends_on", []))
            if deps.issubset(completed):
                wave.append(i)
        if not wave:
            # Circular dependency or bug — force remaining
            wave = [i for i in range(n) if i not in completed]
        waves.append(wave)
        completed.update(wave)

    return waves
