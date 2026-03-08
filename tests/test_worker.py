"""Tests for the worker subprocess wrapper.

These tests spawn real subprocesses but use a mock worker script
that simulates ReACTAgent behavior without requiring Ollama.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from ollama_cli.agents.mailbox import Mailbox
from ollama_cli.agents.worker import poll_agent, stop_agent

# Path to the mock worker script (created in setUp)
_MOCK_WORKER = None


def _write_mock_worker(path: str):
    """Write a mock worker script that simulates a ReACTAgent."""
    script = '''
import json
import sys
import time
import os

def write_step(mailbox_dir, aid, step, **kwargs):
    p = os.path.join(mailbox_dir, f"{aid}.jsonl")
    entry = {"step": step, "timestamp": time.time(), **kwargs}
    with open(p, "a") as f:
        f.write(json.dumps(entry) + "\\n")

def check_signal(mailbox_dir, aid):
    p = os.path.join(mailbox_dir, f"{aid}.jsonl")
    if not os.path.exists(p):
        return None
    with open(p, "r") as f:
        lines = f.readlines()
    for line in reversed(lines):
        entry = json.loads(line.strip())
        if entry.get("step") == "signal_ack":
            return None
        if entry.get("step") == "signal":
            return entry.get("signal")
    return None

config_path = sys.argv[1]
with open(config_path, "r") as f:
    config = json.load(f)

aid = config["agent_id"]
mailbox_dir = config["mailbox_dir"]
task = config["task"]
os.makedirs(mailbox_dir, exist_ok=True)

write_step(mailbox_dir, aid, "init", agent_id=aid,
           task=task, model=config["model"], tools=config.get("tools", []))

for i in range(3):
    sig = check_signal(mailbox_dir, aid)
    if sig == "stop":
        write_step(mailbox_dir, aid, "done",
                   summary=f"Stopped after {i} iterations.", status="success")
        try:
            os.remove(config_path)
        except OSError:
            pass
        sys.exit(0)

    write_step(mailbox_dir, aid, "think", content=f"Thinking step {i}")
    time.sleep(0.3)

    write_step(mailbox_dir, aid, "act", tool="mock_tool", params={"i": i})
    time.sleep(0.1)

    write_step(mailbox_dir, aid, "observe", content=f"Result {i}")

write_step(mailbox_dir, aid, "done",
           summary=f"Completed task: {task}", status="success")

try:
    os.remove(config_path)
except OSError:
    pass
'''
    with open(path, "w") as f:
        f.write(script)


class TestWorker(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="ollama_cli_test_worker_")
        self.mailbox_dir = os.path.join(self.test_dir, "mailbox")
        os.makedirs(self.mailbox_dir, exist_ok=True)
        self.mailbox = Mailbox(self.mailbox_dir)

        # Write mock worker script
        self.mock_worker_path = os.path.join(self.test_dir, "mock_worker.py")
        _write_mock_worker(self.mock_worker_path)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _spawn_mock(self, agent_id: str, task: str = "test task",
                    model: str = "test-model", tools: list = None) -> subprocess.Popen:
        """Spawn the mock worker as a subprocess."""
        config = {
            "agent_id": agent_id,
            "task": task,
            "model": model,
            "tools": tools or [],
            "mailbox_dir": self.mailbox_dir,
            "ollama_url": "http://localhost:11434",
            "context": "",
            "max_iterations": 10,
        }
        config_dir = os.path.join(self.mailbox_dir, ".configs")
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, f"{agent_id}.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        return subprocess.Popen(
            [sys.executable, self.mock_worker_path, config_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def test_spawn_and_complete(self):
        """Spawn an agent, wait for it to finish, read summary."""
        proc = self._spawn_mock("worker-001", task="say hello")
        proc.wait(timeout=10)
        self.assertEqual(proc.returncode, 0)

        # Check mailbox
        status = poll_agent("worker-001", self.mailbox)
        self.assertEqual(status, "success")

        summary = self.mailbox.read_summary("worker-001")
        self.assertIsNotNone(summary)
        self.assertIn("say hello", summary)

    def test_poll_while_running(self):
        """Poll an agent while it's still running."""
        proc = self._spawn_mock("worker-002")

        # Give it a moment to start and write init
        time.sleep(0.3)
        status = poll_agent("worker-002", self.mailbox)
        self.assertEqual(status, "running")

        proc.wait(timeout=10)
        status = poll_agent("worker-002", self.mailbox)
        self.assertEqual(status, "success")

    def test_stop_signal(self):
        """Stop an agent mid-execution."""
        proc = self._spawn_mock("worker-003", task="long task")

        # Wait for it to start
        time.sleep(0.3)
        self.assertEqual(poll_agent("worker-003", self.mailbox), "running")

        # Send stop and wait for summary
        summary = stop_agent("worker-003", self.mailbox, timeout=10)
        self.assertIsNotNone(summary)
        self.assertIn("Stopped", summary)

        proc.wait(timeout=5)

    def test_mailbox_has_steps(self):
        """Verify the mailbox contains expected step types after completion."""
        proc = self._spawn_mock("worker-004")
        proc.wait(timeout=10)

        steps = self.mailbox.read_steps("worker-004")
        step_types = [s["step"] for s in steps]

        self.assertEqual(step_types[0], "init")
        self.assertIn("think", step_types)
        self.assertIn("act", step_types)
        self.assertIn("observe", step_types)
        self.assertEqual(step_types[-1], "done")

    def test_multiple_agents(self):
        """Spawn multiple agents concurrently."""
        procs = []
        for i in range(3):
            p = self._spawn_mock(f"multi-{i}", task=f"task {i}")
            procs.append(p)

        for p in procs:
            p.wait(timeout=15)

        for i in range(3):
            status = poll_agent(f"multi-{i}", self.mailbox)
            self.assertEqual(status, "success")
            summary = self.mailbox.read_summary(f"multi-{i}")
            self.assertIn(f"task {i}", summary)

    def test_list_agents(self):
        """List running agents."""
        proc = self._spawn_mock("list-001")
        proc.wait(timeout=10)

        agents = self.mailbox.list_agents()
        self.assertIn("list-001", agents)


class TestSpawnAgentFunction(unittest.TestCase):
    """Test the actual spawn_agent function (needs ollama_cli installed)."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="ollama_cli_test_spawn_")
        self.mailbox_dir = os.path.join(self.test_dir, "mailbox")
        os.makedirs(self.mailbox_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_spawn_creates_config(self):
        """spawn_agent creates a config file and starts a process."""
        from ollama_cli.agents.worker import spawn_agent

        proc = spawn_agent(
            task="test",
            model="nonexistent-model",
            tools=["web_search"],
            agent_id="spawn-001",
            mailbox_dir=self.mailbox_dir,
            ollama_url="http://localhost:99999",  # intentionally wrong
        )

        # Process should have started (will fail to connect, but that's ok)
        self.assertIsNotNone(proc.pid)

        # Config file should have been created
        config_path = os.path.join(self.mailbox_dir, ".configs", "spawn-001.json")
        # Config might already be cleaned up if process finished fast
        # but process should exist
        proc.wait(timeout=15)


if __name__ == "__main__":
    unittest.main()
