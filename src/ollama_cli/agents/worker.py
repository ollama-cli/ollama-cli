"""
Worker — subprocess wrapper for ReACTAgent.

The parent process spawns a worker via spawn_agent(), which runs
a ReACTAgent in a separate process. Communication happens through
the shared Mailbox (JSONL files on disk).

Parent process API:
    pid = spawn_agent(task, model, tools, agent_id, mailbox_dir, ...)
    status = poll_agent(agent_id, mailbox)
    summary = stop_agent(agent_id, mailbox)

Subprocess entry point:
    python -m ollama_cli.agents.worker <config.json>
"""

import json
import os
import sys
import time
import subprocess
import tempfile
from typing import List, Optional

from ollama_cli.agents.mailbox import Mailbox

# How long stop_agent waits for the agent to summarize and exit
STOP_TIMEOUT = 30


def spawn_agent(task: str, model: str, tools: List[str],
                agent_id: str, mailbox_dir: str,
                ollama_url: str = "http://localhost:11434",
                context: str = "", max_iterations: int = 10) -> subprocess.Popen:
    """Spawn a ReACTAgent in a subprocess.

    Returns the Popen handle. The agent writes to mailbox_dir/<agent_id>.jsonl.
    """
    # Write config to a temp file so we don't hit arg length limits
    config = {
        "agent_id": agent_id,
        "task": task,
        "model": model,
        "tools": tools,
        "mailbox_dir": mailbox_dir,
        "ollama_url": ollama_url,
        "context": context,
        "max_iterations": max_iterations,
    }
    config_dir = os.path.join(mailbox_dir, ".configs")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"{agent_id}.json")
    with open(config_path, "w") as f:
        json.dump(config, f)

    proc = subprocess.Popen(
        [sys.executable, "-m", "ollama_cli.agents.worker", config_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def poll_agent(agent_id: str, mailbox: Mailbox) -> str:
    """Check the status of a spawned agent.

    Returns: "running", "done", "error", "stopped", or "unknown"
    """
    return mailbox.get_status(agent_id)


def stop_agent(agent_id: str, mailbox: Mailbox,
               timeout: int = STOP_TIMEOUT) -> Optional[str]:
    """Send stop signal and wait for the agent to summarize.

    Returns the summary, or None if the agent didn't finish in time.
    """
    mailbox.send_signal(agent_id, "stop")

    deadline = time.time() + timeout
    while time.time() < deadline:
        status = mailbox.get_status(agent_id)
        if status in ("success", "done", "error"):
            return mailbox.read_summary(agent_id)
        time.sleep(0.5)

    return None


def _run_worker(config_path: str):
    """Entry point for the subprocess. Runs the ReACTAgent."""
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Register tools before creating the agent
    import ollama_cli.tools.filesystem
    import ollama_cli.tools.system
    import ollama_cli.tools.web
    import ollama_cli.tools.git
    import ollama_cli.tools.code
    import ollama_cli.tools.knowledge
    import ollama_cli.tools.memory
    import ollama_cli.tools.execution

    from ollama_cli.agents.mailbox import Mailbox
    from ollama_cli.agents.react import ReACTAgent
    from ollama_cli.core.ollama import OllamaClient

    client = OllamaClient(config["ollama_url"])
    mailbox = Mailbox(config["mailbox_dir"])

    agent = ReACTAgent(
        client=client,
        mailbox=mailbox,
        agent_id=config["agent_id"],
        task=config["task"],
        model=config["model"],
        tools=config.get("tools"),
        context=config.get("context", ""),
        max_iterations=config.get("max_iterations", 10),
    )

    try:
        agent.run()
    except Exception as e:
        mailbox.write_step(config["agent_id"], "error", content=str(e))
    finally:
        # Clean up config file
        try:
            os.remove(config_path)
        except OSError:
            pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m ollama_cli.agents.worker <config.json>", file=sys.stderr)
        sys.exit(1)
    _run_worker(sys.argv[1])
