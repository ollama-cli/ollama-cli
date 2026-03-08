"""Tests for the ReACT agent."""

import os
import shutil
import tempfile
import unittest
from unittest import mock

from ollama_cli.agents.mailbox import Mailbox
from ollama_cli.agents.react import ReACTAgent, parse_tool_calls, execute_tool
from ollama_cli.tools.base import registry


class FakeOllamaClient:
    """Mock Ollama client that returns scripted responses."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.call_count = 0

    def chat(self, messages, model, stream=True):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
        else:
            resp = "I have no more responses."
        # Yield the whole response as one chunk (simulates streaming)
        yield resp


class TestParseToolCalls(unittest.TestCase):

    def test_xml_format(self):
        text = """<tool_call>
<tool_name>web_search</tool_name>
<parameters>{"query": "weather in Cork"}</parameters>
</tool_call>"""
        calls = parse_tool_calls(text, ["web_search"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "web_search")
        self.assertEqual(calls[0][1]["query"], "weather in Cork")

    def test_filters_by_allowed_tools(self):
        text = """<tool_call>
<tool_name>web_search</tool_name>
<parameters>{"query": "test"}</parameters>
</tool_call>"""
        # web_search not in allowed list
        calls = parse_tool_calls(text, ["read_file"])
        self.assertEqual(len(calls), 0)

    def test_tool_name_json_format(self):
        # Register a fake tool so it's in the allowed set
        text = 'web_search {"query": "hello"}'
        calls = parse_tool_calls(text, ["web_search"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["query"], "hello")

    def test_no_tool_calls(self):
        calls = parse_tool_calls("The answer is 42.", ["web_search"])
        self.assertEqual(len(calls), 0)

    def test_unclosed_tags(self):
        text = """<tool_call>
<tool_name>web_search</tool_name>
<parameters>{"query": "test"}"""
        calls = parse_tool_calls(text, ["web_search"])
        self.assertEqual(len(calls), 1)


class TestValidateParams(unittest.TestCase):

    def test_simple_params_unchanged(self):
        result = ReACTAgent._validate_params("web_search", {"query": "test", "max_results": 5})
        self.assertEqual(result, {"query": "test", "max_results": 5})

    def test_unwrap_value_key(self):
        """LLM wraps value: {"query": {"value": "test"}}"""
        result = ReACTAgent._validate_params("web_search", {"query": {"value": "test"}})
        self.assertEqual(result["query"], "test")

    def test_unwrap_schema_copy(self):
        """LLM copies schema: {"query": {"type": "string", "description": "...", "value": "test"}}"""
        result = ReACTAgent._validate_params("web_search", {
            "query": {"type": "string", "description": "The search query", "value": "weather Cork"}
        })
        self.assertEqual(result["query"], "weather Cork")

    def test_pure_schema_no_value(self):
        """LLM copies schema without value: {"query": {"type": "string", "description": "..."}}"""
        result = ReACTAgent._validate_params("web_search", {
            "query": {"type": "string", "description": "The search query"}
        })
        # Should be skipped — no usable value
        self.assertNotIn("query", result)

    def test_list_to_string(self):
        result = ReACTAgent._validate_params("test", {"tags": ["a", "b", "c"]})
        self.assertEqual(result["tags"], "a, b, c")

    def test_nested_dict_stringified(self):
        result = ReACTAgent._validate_params("test", {"data": {"foo": "bar"}})
        self.assertIsInstance(result["data"], str)


class TestExecuteTool(unittest.TestCase):

    def test_unknown_tool(self):
        result = execute_tool("nonexistent_tool_xyz", {})
        self.assertIn("Unknown tool", result)

    def test_known_tool(self):
        # run_python is registered — test with a simple expression
        if "run_python" in registry.tools:
            result = execute_tool("run_python", {"code": "print(2+2)"})
            self.assertIn("4", result)


class TestReACTAgent(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="ollama_cli_test_react_")
        self.mailbox = Mailbox(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_direct_answer_no_tools(self):
        """Agent answers directly without using tools."""
        client = FakeOllamaClient(["The answer to 2+2 is 4."])
        agent = ReACTAgent(
            client=client,
            mailbox=self.mailbox,
            agent_id="test-001",
            task="What is 2+2?",
            model="test-model",
            tools=["web_search"],
        )
        summary = agent.run()
        self.assertIsNotNone(summary)
        self.assertIn("4", summary)

        # Verify mailbox steps
        steps = self.mailbox.read_steps("test-001")
        step_types = [s["step"] for s in steps]
        self.assertEqual(step_types[0], "init")
        self.assertIn("think", step_types)
        self.assertIn("done", step_types)

        # Verify summary in mailbox
        mb_summary = self.mailbox.read_summary("test-001")
        self.assertIsNotNone(mb_summary)
        self.assertIn("4", mb_summary)

    def test_tool_call_then_answer(self):
        """Agent calls a tool, gets result, then answers."""
        # First response: tool call
        tool_response = """I need to search for this.
<tool_call>
<tool_name>web_search</tool_name>
<parameters>{"query": "weather Cork"}</parameters>
</tool_call>"""
        # Second response: final answer after seeing tool result
        final_response = "The weather in Cork is 9°C and cloudy."

        client = FakeOllamaClient([tool_response, final_response])
        agent = ReACTAgent(
            client=client,
            mailbox=self.mailbox,
            agent_id="test-002",
            task="What is the weather in Cork?",
            model="test-model",
            tools=["web_search"],
        )

        with mock.patch("ollama_cli.agents.react.execute_tool",
                        return_value="Cork: 9°C, cloudy"):
            summary = agent.run()

        self.assertIsNotNone(summary)
        self.assertIn("9°C", summary)

        steps = self.mailbox.read_steps("test-002")
        step_types = [s["step"] for s in steps]
        self.assertIn("think", step_types)
        self.assertIn("act", step_types)
        self.assertIn("observe", step_types)
        self.assertIn("done", step_types)

        # Verify act step has tool info
        act_steps = [s for s in steps if s["step"] == "act"]
        self.assertEqual(act_steps[0]["tool"], "web_search")

    def test_stop_signal(self):
        """Agent stops when it sees a stop signal mid-execution."""
        # First response triggers a tool call, giving us a chance to inject stop
        tool_response = """<tool_call>
<tool_name>web_search</tool_name>
<parameters>{"query": "long task"}</parameters>
</tool_call>"""
        client = FakeOllamaClient([tool_response, "Should not reach this."])
        agent = ReACTAgent(
            client=client,
            mailbox=self.mailbox,
            agent_id="test-003",
            task="Do something long",
            model="test-model",
            tools=["web_search"],
        )

        def fake_execute_and_signal(name, params):
            # Inject stop signal after the tool executes but before next iteration
            self.mailbox.send_signal("test-003", "stop")
            return "partial result"

        with mock.patch("ollama_cli.agents.react.execute_tool",
                        side_effect=fake_execute_and_signal):
            summary = agent.run()

        self.assertIsNotNone(summary)
        self.assertIn("Stopped", summary)

    def test_duplicate_tool_calls_skipped(self):
        """Agent doesn't execute the same tool call twice."""
        same_call = """<tool_call>
<tool_name>web_search</tool_name>
<parameters>{"query": "test"}</parameters>
</tool_call>"""
        client = FakeOllamaClient([same_call, same_call, "Done."])
        agent = ReACTAgent(
            client=client,
            mailbox=self.mailbox,
            agent_id="test-004",
            task="Search test",
            model="test-model",
            tools=["web_search"],
        )

        call_count = [0]
        original_execute = execute_tool

        def counting_execute(name, params):
            call_count[0] += 1
            return "result"

        with mock.patch("ollama_cli.agents.react.execute_tool", side_effect=counting_execute):
            summary = agent.run()

        # Tool should only be called once despite two identical calls
        self.assertEqual(call_count[0], 1)

    def test_max_iterations(self):
        """Agent stops after max iterations."""
        # Always return a tool call — should hit max iterations
        tool_call = """<tool_call>
<tool_name>web_search</tool_name>
<parameters>{"query": "iteration {i}"}</parameters>
</tool_call>"""

        # Create enough unique tool calls to not hit dedup
        responses = []
        for i in range(5):
            responses.append(f"""<tool_call>
<tool_name>web_search</tool_name>
<parameters>{{"query": "iter_{i}"}}</parameters>
</tool_call>""")

        client = FakeOllamaClient(responses)
        agent = ReACTAgent(
            client=client,
            mailbox=self.mailbox,
            agent_id="test-005",
            task="Infinite loop test",
            model="test-model",
            tools=["web_search"],
            max_iterations=3,
        )

        with mock.patch("ollama_cli.agents.react.execute_tool", return_value="ok"):
            summary = agent.run()

        self.assertIn("maximum iterations", summary)

        # Verify status in mailbox
        status = self.mailbox.get_status("test-005")
        self.assertEqual(status, "success")

    def test_mailbox_has_correct_structure(self):
        """Verify the full mailbox trace has expected structure."""
        tool_resp = """<tool_call>
<tool_name>web_search</tool_name>
<parameters>{"query": "hello"}</parameters>
</tool_call>"""
        client = FakeOllamaClient([tool_resp, "The answer is hello."])
        agent = ReACTAgent(
            client=client,
            mailbox=self.mailbox,
            agent_id="test-006",
            task="Say hello",
            model="test-model",
            tools=["web_search"],
        )

        with mock.patch("ollama_cli.agents.react.execute_tool", return_value="hello world"):
            agent.run()

        steps = self.mailbox.read_steps("test-006")

        # Every step must have a timestamp
        for step in steps:
            self.assertIn("timestamp", step)
            self.assertIsInstance(step["timestamp"], float)

        # First step is init
        self.assertEqual(steps[0]["step"], "init")
        self.assertEqual(steps[0]["task"], "Say hello")

        # Last step is done with summary
        self.assertEqual(steps[-1]["step"], "done")
        self.assertIn("summary", steps[-1])
        self.assertEqual(steps[-1]["status"], "success")


if __name__ == "__main__":
    unittest.main()
