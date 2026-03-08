"""Tests for the planner module."""

import json
import unittest
from unittest import mock

from ollama_cli.agents.planner import plan


class FakeClient:
    def __init__(self, response):
        self.response = response

    def chat(self, messages, model, stream=True):
        yield self.response


class TestPlanner(unittest.TestCase):

    def test_simple_task_returns_none(self):
        client = FakeClient("DIRECT")
        result = plan(client, "test-model", "what is 2+2?")
        self.assertIsNone(result)

    def test_direct_case_insensitive(self):
        client = FakeClient("direct")
        result = plan(client, "test-model", "hello")
        self.assertIsNone(result)

    def test_complex_task_returns_subtasks(self):
        response = '''[
  {"task": "Search for weather in Cork", "tools": ["web_search"]},
  {"task": "Search for weather in Dublin", "tools": ["web_search"]}
]'''
        client = FakeClient(response)
        result = plan(client, "test-model", "compare weather in Cork and Dublin")
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertIn("Cork", result[0]["task"])
        self.assertIn("Dublin", result[1]["task"])
        self.assertEqual(result[0]["tools"], ["web_search"])

    def test_json_in_markdown(self):
        response = '''```json
[
  {"task": "Find population of Paris", "tools": ["web_search"]},
  {"task": "Find population of London", "tools": ["web_search"]}
]
```'''
        client = FakeClient(response)
        result = plan(client, "test-model", "compare populations")
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    def test_single_subtask_returns_none(self):
        """Don't delegate if only 1 subtask (just do it directly)."""
        response = '[{"task": "search for X", "tools": ["web_search"]}]'
        client = FakeClient(response)
        result = plan(client, "test-model", "search for X")
        self.assertIsNone(result)

    def test_max_4_subtasks(self):
        tasks = [{"task": f"task {i}", "tools": ["web_search"]} for i in range(6)]
        client = FakeClient(json.dumps(tasks))
        result = plan(client, "test-model", "do many things")
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result), 4)

    def test_invalid_json_returns_none(self):
        client = FakeClient("Here is my plan: task 1, task 2")
        result = plan(client, "test-model", "do stuff")
        self.assertIsNone(result)

    def test_missing_task_field(self):
        response = '[{"description": "no task field", "tools": ["web_search"]}, {"description": "also no task"}]'
        client = FakeClient(response)
        result = plan(client, "test-model", "test")
        self.assertIsNone(result)

    def test_default_tools(self):
        response = '[{"task": "find X"}, {"task": "find Y"}]'
        client = FakeClient(response)
        result = plan(client, "test-model", "find X and Y")
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["tools"], ["web_search"])

    def test_api_error_returns_none(self):
        class ErrorClient:
            def chat(self, messages, model, stream=True):
                raise ConnectionError("no connection")
        result = plan(ErrorClient(), "model", "test")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
