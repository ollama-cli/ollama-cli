"""
ReACT Agent — Reason + Act state machine for subagents.

Runs a THINK → ACT → OBSERVE loop, writing each step to a Mailbox.
On completion or stop signal, self-summarizes and writes a "done" step.
"""

import json
import re
import logging
from typing import List, Dict, Optional

from ollama_cli.agents.mailbox import Mailbox
from ollama_cli.core.ollama import OllamaClient
from ollama_cli.tools.base import registry

logger = logging.getLogger("ollama-cli.react")

# Maximum ReACT iterations before forcing a summary
MAX_ITERATIONS = 10


def parse_tool_calls(text: str, allowed_tools: List[str] = None) -> List[tuple]:
    """Parse tool calls from LLM output.

    Supports multiple formats: <tool_call> XML blocks, <tool_name> tags,
    tool_name {json}, and naked key=value heuristics.

    If allowed_tools is provided, only those tools are recognized.
    """
    tool_names = set(allowed_tools) if allowed_tools else set(registry.tools.keys())
    calls = []

    # 1. <tool_call> blocks
    tc_blocks = re.findall(r'<tool_call>(.*?)(?:</tool_call>|$)', text, re.DOTALL)
    if tc_blocks:
        for block in tc_blocks:
            if not block.strip():
                continue
            name = None
            name_match = re.search(r'<tool_name>(.*?)(?:</tool_name>|$)', block, re.DOTALL)
            if name_match:
                name = name_match.group(1).strip()
            else:
                first_line = block.strip().split('\n')[0].strip()
                if first_line in tool_names:
                    name = first_line

            params_match = re.search(r'<parameters>(.*?)(?:</parameters>|$)', block, re.DOTALL)
            if name and name in tool_names and params_match:
                try:
                    params_str = params_match.group(1).strip()
                    if params_str.startswith('{') and not params_str.endswith('}'):
                        params_str += '}'
                    calls.append((name, json.loads(params_str)))
                except Exception:
                    json_match = re.search(r'(\{.*\})', params_str, re.DOTALL)
                    if json_match:
                        try:
                            calls.append((name, json.loads(json_match.group(1))))
                        except Exception:
                            pass

    # 2. <tool_name> tags or <tool_name> {json} </tool_name>
    if not calls:
        for tool_name in tool_names:
            attr_pattern = fr'<{tool_name}\s+([^>]*?)[\s/]*>'
            for attr_str in re.findall(attr_pattern, text, re.DOTALL):
                attrs = {}
                for k, _, v in re.findall(r'(\w+)\s*=\s*(["\'])(.*?)\2', attr_str):
                    attrs[k] = v
                if attrs:
                    calls.append((tool_name, attrs))

            block_pattern = fr'<{tool_name}>(.*?)(?:</{tool_name}>|$)'
            for b in re.findall(block_pattern, text, re.DOTALL):
                json_match = re.search(r'(\{.*\})', b, re.DOTALL)
                if json_match:
                    try:
                        calls.append((tool_name, json.loads(json_match.group(1))))
                    except Exception:
                        pass

    # 3. tool_name {json}
    if not calls:
        for tool_name in tool_names:
            for json_str in re.findall(fr'{tool_name}\s*({{.*?}})', text, re.DOTALL):
                try:
                    calls.append((tool_name, json.loads(json_str.strip())))
                except Exception:
                    pass

    # 4. <tool_name>key="value"...</tool_name> or <tool_name key="value"... />
    #    Handles sloppy LLM output like <web_search>query="test"max_results=5</web_search>
    if not calls:
        for tool_name in tool_names:
            # Body contains key=value pairs
            body_pattern = fr'<{tool_name}>(.*?)(?:</{tool_name}>|$)'
            for body in re.findall(body_pattern, text, re.DOTALL):
                attrs = {}
                # Quoted: key="value" or key='value'
                for k, _, v in re.findall(r'(\w+)\s*=\s*(["\'])(.*?)\2', body):
                    attrs[k] = v
                # Also capture unquoted: key=value (won't overwrite quoted ones)
                for k, v in re.findall(r'(\w+)\s*=\s*([^\s"\'<>=]+)', body):
                    if k not in attrs:
                        attrs[k] = v
                if attrs:
                    # Convert numeric strings
                    for k, v in attrs.items():
                        if v.isdigit():
                            attrs[k] = int(v)
                    calls.append((tool_name, attrs))

            # Opening tag contains attributes: <tool_name key="value" ...>
            attr_tag = fr'<{tool_name}\s+([^>]*?)/?>'
            for attr_str in re.findall(attr_tag, text, re.DOTALL):
                attrs = {}
                for k, _, v in re.findall(r'(\w+)\s*=\s*(["\'])(.*?)\2', attr_str):
                    attrs[k] = v
                if attrs:
                    for k, v in attrs.items():
                        if isinstance(v, str) and v.isdigit():
                            attrs[k] = int(v)
                    calls.append((tool_name, attrs))

    # 5. Nested XML: <tool_name><param>value</param>...</tool_name>
    if not calls:
        for tool_name in tool_names:
            body_pattern = fr'<{tool_name}>\s*(.*?)\s*</{tool_name}>'
            for body in re.findall(body_pattern, text, re.DOTALL):
                # Look for <key>value</key> pairs inside the body
                params = {}
                for k, v in re.findall(r'<(\w+)>(.*?)</\1>', body, re.DOTALL):
                    v = v.strip()
                    if v.isdigit():
                        params[k] = int(v)
                    else:
                        params[k] = v
                if params:
                    calls.append((tool_name, params))

    return calls


def execute_tool(name: str, params: Dict) -> str:
    """Execute a registered tool by name."""
    tool = registry.get_tool(name)
    if not tool:
        return f"Unknown tool: {name}"
    try:
        return str(tool.execute(**params))
    except Exception as e:
        return f"Error executing {name}: {e}"


class ReACTAgent:
    """A ReACT (Reason + Act) agent that runs in-process.

    Usage (called by the orchestrator):
        agent = ReACTAgent(
            client=ollama_client,
            mailbox=session_mailbox,
            agent_id="agent-001",
            task="What is the weather in Cork?",
            model="mistral:latest",
            tools=["web_search"],
            context="Focus on current temperature and 5-day forecast.",
        )
        summary = agent.run()
    """

    def __init__(self, client: OllamaClient, mailbox: Mailbox,
                 agent_id: str, task: str, model: str,
                 tools: List[str] = None, context: str = "",
                 max_iterations: int = MAX_ITERATIONS):
        """
        Args:
            client: OllamaClient instance
            mailbox: Mailbox instance (per-session)
            agent_id: Unique ID for this agent
            task: The task prompt from the orchestrator
            model: Which Ollama model to use
            tools: List of tool names this agent is allowed to use.
                   The orchestrator decides which tools are relevant.
            context: Extra instructions/context from the orchestrator.
                     Injected into the system prompt (e.g. "focus on
                     Irish weather sources", "output as JSON", etc.)
            max_iterations: Max ReACT loops before forced summary
        """
        self.client = client
        self.mailbox = mailbox
        self.agent_id = agent_id
        self.task = task
        self.model = model
        self.allowed_tools = tools or list(registry.tools.keys())
        self.context = context
        self.max_iterations = max_iterations
        self.messages = []
        self.executed_calls = set()

    def _build_system_prompt(self) -> str:
        tool_desc = ""
        for name in self.allowed_tools:
            tool = registry.get_tool(name)
            if tool:
                tool_desc += f"- {name}: {tool.description}\n"
                tool_desc += f"  Parameters: {json.dumps(tool.parameters)}\n"

        context_block = f"\nAdditional instructions from orchestrator:\n{self.context}\n" if self.context else ""

        return f"""You are a focused agent working on a specific task.
You have access to tools to help you. When you need a tool, use this format:

<tool_call>
<tool_name>name</tool_name>
<parameters>{{"param": "value"}}</parameters>
</tool_call>

Available tools:
{tool_desc}
{context_block}
RULES:
1. Think step by step about how to complete the task.
2. Use tools when needed, one at a time.
3. After getting a tool result, reason about it before acting again.
4. When you have the final answer, just state it clearly — do NOT call any more tools.
5. Be concise."""

    def run(self) -> Optional[str]:
        """Run the ReACT loop. Returns the final summary or None on error."""
        # Initialize mailbox
        self.mailbox.create(self.agent_id, self.task, self.model, self.allowed_tools)

        # Build initial messages
        self.messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self.task},
        ]

        for iteration in range(self.max_iterations):
            # Check for stop signal
            signal = self.mailbox.check_signal(self.agent_id)
            if signal == "stop":
                return self._summarize("Stopped by orchestrator before completion.")

            # THINK — call LLM
            try:
                response = self._call_llm()
            except Exception as e:
                self.mailbox.write_step(self.agent_id, "error", content=str(e))
                return None

            self.messages.append({"role": "assistant", "content": response})
            self.mailbox.write_step(self.agent_id, "think", content=response)

            # Parse tool calls from response
            tool_calls = parse_tool_calls(response, self.allowed_tools)

            if not tool_calls:
                # No tool calls — the agent is done, response IS the answer
                return self._summarize(response)

            # ACT + OBSERVE for each tool call
            results = []
            for name, params in tool_calls:
                call_key = f"{name}:{json.dumps(params, sort_keys=True)}"
                if call_key in self.executed_calls:
                    continue
                self.executed_calls.add(call_key)

                self.mailbox.write_step(self.agent_id, "act",
                                        tool=name, params=params)

                result = execute_tool(name, params)

                self.mailbox.write_step(self.agent_id, "observe",
                                        tool=name, content=result[:2000])
                results.append(f"Tool Result ({name}):\n{result}")

            if not results:
                # All tool calls were duplicates
                return self._summarize(response)

            # Feed results back to the LLM
            feedback = "\n\n".join(results)
            self.messages.append({"role": "user", "content": feedback})

            # Check stop signal again after tool execution
            signal = self.mailbox.check_signal(self.agent_id)
            if signal == "stop":
                return self._summarize("Stopped by orchestrator during execution.")

        # Hit max iterations
        return self._summarize("Reached maximum iterations without a final answer.")

    def _call_llm(self) -> str:
        """Call Ollama and collect the full response."""
        full_response = ""
        for chunk in self.client.chat(self.messages, self.model):
            full_response += chunk
        return full_response

    def _summarize(self, final_content: str) -> str:
        """Ask the LLM to produce a concise summary, write it to mailbox."""
        # If the content is already short, use it directly
        if len(final_content) < 500:
            summary = final_content
        else:
            # Ask the model to summarize
            try:
                summary_messages = self.messages + [
                    {"role": "user", "content":
                     "Summarize your findings in 2-3 sentences. "
                     "Include only the final answer, not your reasoning steps. "
                     "Do NOT call any tools."}
                ]
                summary = ""
                for chunk in self.client.chat(summary_messages, self.model):
                    summary += chunk
            except Exception:
                # Fallback: truncate
                summary = final_content[:500]

        self.mailbox.write_step(self.agent_id, "done",
                                summary=summary, status="success")
        return summary
