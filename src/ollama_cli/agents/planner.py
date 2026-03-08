"""
Planner — decides whether to handle a task directly or delegate to subagents.

The orchestrator calls plan() before processing a user prompt. If the task
can benefit from parallel subagents, it returns a list of subtasks.
Otherwise it returns None and the orchestrator handles it directly.
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

If the task is COMPLEX and can be broken into INDEPENDENT subtasks that can run in parallel, respond with a JSON array of subtasks. Each subtask must have:
- "task": what the subagent should do (a clear, self-contained instruction)
- "tools": list of tool names the subagent needs (from: web_search, read_file, write_file, list_directory, grep_search, run_python, code_analyze_file)

Example for "compare weather in Cork and Dublin":
[
  {"task": "Search for the current weather in Cork, Ireland", "tools": ["web_search"]},
  {"task": "Search for the current weather in Dublin, Ireland", "tools": ["web_search"]}
]

Example for "what is 2+2":
DIRECT

RULES:
- Only delegate if subtasks are truly INDEPENDENT (can run in parallel)
- Do NOT delegate simple questions, greetings, or single-step tasks
- Keep subtasks focused — each should be answerable by one agent
- Maximum 4 subtasks
- Respond with ONLY "DIRECT" or a JSON array, nothing else"""


def plan(client: OllamaClient, model: str, user_input: str) -> Optional[List[dict]]:
    """Decide whether to delegate a task to subagents.

    Returns a list of subtask dicts [{"task": ..., "tools": [...]}, ...],
    or None if the task should be handled directly.
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

    # Check for DIRECT
    if response.upper().startswith("DIRECT"):
        return None

    # Try to parse JSON array
    try:
        # Extract JSON array from response (LLM might wrap it in markdown)
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            subtasks = json.loads(json_match.group())
            if isinstance(subtasks, list) and len(subtasks) > 0:
                # Validate structure
                valid = []
                for st in subtasks[:4]:  # max 4 subtasks
                    if isinstance(st, dict) and "task" in st:
                        valid.append({
                            "task": str(st["task"]),
                            "tools": st.get("tools", ["web_search"]),
                        })
                if len(valid) >= 2:  # only delegate if 2+ subtasks
                    return valid
    except (json.JSONDecodeError, ValueError):
        pass

    # Couldn't parse — handle directly
    return None
