"""Tests for the planner module."""

import json
import unittest
from unittest import mock

from ollama_cli.agents.planner import plan, get_execution_waves


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


class TestDependencies(unittest.TestCase):

    def test_dependencies_parsed(self):
        response = json.dumps([
            {"task": "Find capital of France", "tools": ["web_search"]},
            {"task": "Find population of the capital", "tools": ["web_search"], "depends_on": [0]}
        ])
        client = FakeClient(response)
        result = plan(client, "model", "population of France's capital")
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["depends_on"], [])
        self.assertEqual(result[1]["depends_on"], [0])

    def test_invalid_dep_index_filtered(self):
        """depends_on referencing a future or out-of-range index is dropped."""
        response = json.dumps([
            {"task": "A", "depends_on": [5]},
            {"task": "B", "depends_on": [0, 99]}
        ])
        client = FakeClient(response)
        result = plan(client, "model", "test")
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["depends_on"], [])
        self.assertEqual(result[1]["depends_on"], [0])

    def test_no_self_dependency(self):
        """Task cannot depend on itself."""
        response = json.dumps([
            {"task": "A"},
            {"task": "B", "depends_on": [1]}  # self-reference
        ])
        client = FakeClient(response)
        result = plan(client, "model", "test")
        self.assertEqual(result[1]["depends_on"], [])


class TestExecutionWaves(unittest.TestCase):

    def test_all_parallel(self):
        subtasks = [
            {"task": "A", "depends_on": []},
            {"task": "B", "depends_on": []},
            {"task": "C", "depends_on": []},
        ]
        waves = get_execution_waves(subtasks)
        self.assertEqual(len(waves), 1)
        self.assertEqual(sorted(waves[0]), [0, 1, 2])

    def test_simple_chain(self):
        subtasks = [
            {"task": "A", "depends_on": []},
            {"task": "B", "depends_on": [0]},
        ]
        waves = get_execution_waves(subtasks)
        self.assertEqual(len(waves), 2)
        self.assertEqual(waves[0], [0])
        self.assertEqual(waves[1], [1])

    def test_diamond(self):
        """A → B, A → C, B+C → D"""
        subtasks = [
            {"task": "A", "depends_on": []},
            {"task": "B", "depends_on": [0]},
            {"task": "C", "depends_on": [0]},
            {"task": "D", "depends_on": [1, 2]},
        ]
        waves = get_execution_waves(subtasks)
        self.assertEqual(len(waves), 3)
        self.assertEqual(waves[0], [0])
        self.assertIn(1, waves[1])
        self.assertIn(2, waves[1])
        self.assertEqual(waves[2], [3])

    def test_mixed_parallel_and_deps(self):
        subtasks = [
            {"task": "A", "depends_on": []},
            {"task": "B", "depends_on": []},
            {"task": "C", "depends_on": [0]},
        ]
        waves = get_execution_waves(subtasks)
        self.assertEqual(len(waves), 2)
        self.assertIn(0, waves[0])
        self.assertIn(1, waves[0])
        self.assertEqual(waves[1], [2])


if __name__ == "__main__":
    unittest.main()
