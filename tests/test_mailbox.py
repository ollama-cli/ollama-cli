"""Tests for the Mailbox class."""

import os
import shutil
import tempfile
import unittest

from ollama_cli.agents.mailbox import Mailbox


class TestMailbox(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="ollama_cli_test_mailbox_")
        self.mb = Mailbox(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_create_mailbox(self):
        path = self.mb.create("agent-001", "search weather", "mistral:latest", ["web_search"])
        self.assertTrue(os.path.exists(path))

        steps = self.mb.read_steps("agent-001")
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["step"], "init")
        self.assertEqual(steps[0]["task"], "search weather")
        self.assertEqual(steps[0]["model"], "mistral:latest")
        self.assertEqual(steps[0]["tools"], ["web_search"])

    def test_write_and_read_steps(self):
        self.mb.create("agent-002", "test task", "llama3.2:3b")
        self.mb.write_step("agent-002", "think", content="I need to search")
        self.mb.write_step("agent-002", "act", tool="web_search", params={"query": "test"})
        self.mb.write_step("agent-002", "observe", content="Found results")
        self.mb.write_step("agent-002", "done", summary="Test completed.", status="success")

        steps = self.mb.read_steps("agent-002")
        self.assertEqual(len(steps), 5)  # init + 4 steps
        self.assertEqual(steps[1]["step"], "think")
        self.assertEqual(steps[2]["step"], "act")
        self.assertEqual(steps[2]["tool"], "web_search")
        self.assertEqual(steps[3]["step"], "observe")
        self.assertEqual(steps[4]["step"], "done")

    def test_read_summary_when_done(self):
        self.mb.create("agent-003", "task", "model")
        self.mb.write_step("agent-003", "think", content="thinking")
        self.mb.write_step("agent-003", "done", summary="The answer is 42.", status="success")

        summary = self.mb.read_summary("agent-003")
        self.assertEqual(summary, "The answer is 42.")

    def test_read_summary_when_not_done(self):
        self.mb.create("agent-004", "task", "model")
        self.mb.write_step("agent-004", "think", content="still thinking")

        summary = self.mb.read_summary("agent-004")
        self.assertIsNone(summary)

    def test_read_summary_nonexistent_agent(self):
        summary = self.mb.read_summary("agent-nonexistent")
        self.assertIsNone(summary)

    def test_get_status(self):
        # Unknown
        self.assertEqual(self.mb.get_status("agent-ghost"), "unknown")

        # Running
        self.mb.create("agent-005", "task", "model")
        self.mb.write_step("agent-005", "think", content="working")
        self.assertEqual(self.mb.get_status("agent-005"), "running")

        # Done
        self.mb.write_step("agent-005", "done", summary="done", status="success")
        self.assertEqual(self.mb.get_status("agent-005"), "success")

    def test_get_status_error(self):
        self.mb.create("agent-006", "task", "model")
        self.mb.write_step("agent-006", "error", content="something broke")
        self.assertEqual(self.mb.get_status("agent-006"), "error")

    def test_send_and_check_signal(self):
        self.mb.create("agent-007", "task", "model")
        self.mb.write_step("agent-007", "think", content="working")

        # No signal yet
        self.assertIsNone(self.mb.check_signal("agent-007"))

        # Send stop
        self.mb.send_signal("agent-007", "stop")
        self.assertEqual(self.mb.check_signal("agent-007"), "stop")

    def test_signal_cleared_by_new_step(self):
        self.mb.create("agent-008", "task", "model")
        self.mb.send_signal("agent-008", "stop")
        self.assertEqual(self.mb.check_signal("agent-008"), "stop")

        # Agent writes a new step after seeing the signal
        self.mb.write_step("agent-008", "think", content="acknowledged stop, summarizing")
        # Signal should no longer be visible (new step is after it)
        self.assertIsNone(self.mb.check_signal("agent-008"))

    def test_list_agents(self):
        self.mb.create("agent-a", "task1", "model")
        self.mb.create("agent-b", "task2", "model")
        agents = self.mb.list_agents()
        self.assertIn("agent-a", agents)
        self.assertIn("agent-b", agents)
        self.assertEqual(len(agents), 2)

    def test_cleanup_with_archive(self):
        self.mb.create("agent-009", "task", "model")
        self.mb.write_step("agent-009", "done", summary="done", status="success")
        path = self.mb._path("agent-009")
        self.assertTrue(os.path.exists(path))

        self.mb.cleanup("agent-009", archive=True)
        self.assertFalse(os.path.exists(path))

        # Should be in archive
        archive_files = os.listdir(self.mb.archive_dir)
        self.assertEqual(len(archive_files), 1)
        self.assertTrue(archive_files[0].startswith("agent-009_"))

    def test_cleanup_without_archive(self):
        self.mb.create("agent-010", "task", "model")
        path = self.mb._path("agent-010")
        self.assertTrue(os.path.exists(path))

        self.mb.cleanup("agent-010", archive=False)
        self.assertFalse(os.path.exists(path))

    def test_cleanup_nonexistent(self):
        # Should not raise
        self.mb.cleanup("agent-nonexistent")

    def test_cleanup_all(self):
        self.mb.create("agent-x", "task", "model")
        self.mb.create("agent-y", "task", "model")
        self.assertEqual(len(self.mb.list_agents()), 2)

        self.mb.cleanup_all(archive=False)
        self.assertEqual(len(self.mb.list_agents()), 0)

    def test_steps_have_timestamps(self):
        self.mb.create("agent-011", "task", "model")
        self.mb.write_step("agent-011", "think", content="test")

        steps = self.mb.read_steps("agent-011")
        for step in steps:
            self.assertIn("timestamp", step)
            self.assertIsInstance(step["timestamp"], float)

    def test_concurrent_writes(self):
        """Verify multiple writes don't corrupt the file."""
        self.mb.create("agent-012", "task", "model")
        for i in range(50):
            self.mb.write_step("agent-012", "think", content=f"step {i}")

        steps = self.mb.read_steps("agent-012")
        self.assertEqual(len(steps), 51)  # init + 50

    def test_separate_mailboxes_are_isolated(self):
        """Two Mailbox instances with different dirs don't interfere."""
        other_dir = self.test_dir + "_other"
        os.makedirs(other_dir, exist_ok=True)
        try:
            other_mb = Mailbox(other_dir)
            self.mb.create("agent-x", "task1", "model")
            other_mb.create("agent-x", "task2", "model")

            steps1 = self.mb.read_steps("agent-x")
            steps2 = other_mb.read_steps("agent-x")
            self.assertEqual(steps1[0]["task"], "task1")
            self.assertEqual(steps2[0]["task"], "task2")
        finally:
            shutil.rmtree(other_dir)


if __name__ == "__main__":
    unittest.main()
