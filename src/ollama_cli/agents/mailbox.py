"""
Mailbox — file-based message passing for subagents.

Each session owns a Mailbox instance whose files live inside the session
directory: ~/.ollama-cli/sessions/<session_id>/mailbox/

Each agent gets a JSONL file: <mailbox_dir>/<agent_id>.jsonl
Steps are appended as JSON lines. The orchestrator reads only the
final summary; full traces stay on disk for debugging.
"""

import json
import os
import time
import shutil
from typing import Optional


class Mailbox:
    """Per-session mailbox for subagent communication."""

    def __init__(self, base_dir: str):
        """Create a mailbox rooted at base_dir.

        Typically: ~/.ollama-cli/sessions/<session_id>/mailbox
        """
        self.base_dir = base_dir
        self.archive_dir = os.path.join(base_dir, "archive")

    def _ensure_dirs(self):
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)

    def _path(self, agent_id: str) -> str:
        return os.path.join(self.base_dir, f"{agent_id}.jsonl")

    def create(self, agent_id: str, task: str, model: str, tools: list = None) -> str:
        """Create a new mailbox file for an agent. Returns the file path."""
        self._ensure_dirs()
        path = self._path(agent_id)
        header = {
            "step": "init",
            "agent_id": agent_id,
            "task": task,
            "model": model,
            "tools": tools or [],
            "timestamp": time.time(),
        }
        with open(path, "w") as f:
            f.write(json.dumps(header) + "\n")
        return path

    def write_step(self, agent_id: str, step: str, **kwargs):
        """Append a step to an agent's mailbox.

        step: one of "think", "act", "observe", "done", "error"
        kwargs: content, tool, params, summary, status, etc.
        """
        path = self._path(agent_id)
        entry = {"step": step, "timestamp": time.time(), **kwargs}
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def read_steps(self, agent_id: str) -> list:
        """Read all steps from an agent's mailbox."""
        path = self._path(agent_id)
        if not os.path.exists(path):
            return []
        steps = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        steps.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return steps

    def read_summary(self, agent_id: str) -> Optional[str]:
        """Read only the final summary from an agent's mailbox.

        Returns None if the agent hasn't finished yet.
        """
        steps = self.read_steps(agent_id)
        for step in reversed(steps):
            if step.get("step") == "done":
                return step.get("summary")
        return None

    def get_status(self, agent_id: str) -> str:
        """Get the current status of an agent.

        Returns: "unknown", "running", "done", "error", "stopped"
        """
        steps = self.read_steps(agent_id)
        if not steps:
            return "unknown"
        last = steps[-1]
        last_step = last.get("step", "")
        if last_step == "done":
            return last.get("status", "done")
        if last_step == "error":
            return "error"
        if last.get("signal") == "stop":
            return "stopped"
        return "running"

    def send_signal(self, agent_id: str, signal: str):
        """Send a signal to a running agent (e.g. 'stop')."""
        self.write_step(agent_id, "signal", signal=signal)

    def check_signal(self, agent_id: str) -> Optional[str]:
        """Check if there's a pending signal for this agent.

        Called by the subagent between ReACT steps.
        Returns the signal string (e.g. 'stop') or None.
        """
        steps = self.read_steps(agent_id)
        for step in reversed(steps):
            if step.get("step") == "signal":
                return step.get("signal")
            if step.get("step") in ("think", "act", "observe", "init"):
                break
        return None

    def list_agents(self) -> list:
        """List all agent IDs with active mailbox files."""
        self._ensure_dirs()
        agents = []
        for f in os.listdir(self.base_dir):
            if f.endswith(".jsonl"):
                agents.append(f[:-6])
        return agents

    def cleanup(self, agent_id: str, archive: bool = True):
        """Remove an agent's mailbox. Optionally archive it first."""
        path = self._path(agent_id)
        if not os.path.exists(path):
            return
        if archive:
            self._ensure_dirs()
            dest = os.path.join(self.archive_dir, f"{agent_id}_{int(time.time())}.jsonl")
            shutil.move(path, dest)
        else:
            os.remove(path)

    def cleanup_all(self, archive: bool = False):
        """Remove all mailbox files."""
        for agent_id in self.list_agents():
            self.cleanup(agent_id, archive=archive)
