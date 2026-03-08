"""
Ollama CLI v3.0
License: MIT
Author: Gemini CLI
"""

import sys
import os
import json
import re
import signal
import logging
from typing import List, Dict, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.expanduser("~/.ollama-cli.log")),
    ]
)
logger = logging.getLogger("ollama-cli")

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ollama_cli.core.config import load_config, save_config, update_config
from ollama_cli.core.ollama import OllamaClient
from ollama_cli.core.sessions import save_session, load_session, list_sessions, generate_session_id, get_session_preview, get_mailbox_dir
from ollama_cli.agents.mailbox import Mailbox
from ollama_cli.agents.worker import spawn_agent, poll_agent, stop_agent
from ollama_cli.agents.planner import plan as plan_subtasks, get_execution_waves
from ollama_cli.ui.formatter import console, print_markdown, print_error, print_status, StreamingDisplay
from prompt_toolkit.formatted_text import HTML
from ollama_cli.ui.repl import REPL, CheckpointRestore
from ollama_cli.tools.base import registry
import ollama_cli.tools.filesystem
import ollama_cli.tools.system
import ollama_cli.tools.web
import ollama_cli.tools.git
import ollama_cli.tools.code
import ollama_cli.tools.knowledge
import ollama_cli.tools.memory
import ollama_cli.tools.execution
import ollama_cli.tools.media
from ollama_cli.integrations.mcp import connect_mcp_server, call_mcp_tool, MCP_TOOLS
from ollama_cli.integrations.lsp import start_lsp_server, lsp_goto_definition
from ollama_cli.integrations.notify import start_notify_thread, send_notification

VERSION = "3.0.0"

class OllamaCLI:
    def __init__(self, session_id: str = None):
        self.config = load_config()
        self.client = OllamaClient(self.config.get("ollama_url"))
        self.repl = REPL()
        self.messages = []
        self.available_models = []
        self.current_model = self.config.get("default_model", "llama3.2")
        self.auto_model = True
        self.system_prompt = self._build_system_prompt()
        self.max_tool_iterations = 5
        self.checkpoints = []  # list of (messages_copy, model, prompt_text) tuples
        self.max_checkpoints = 5
        self._resume_id = session_id  # set if --session was passed
        self.session_id = session_id or generate_session_id()
        self.mailbox = Mailbox(get_mailbox_dir(self.session_id))
        self._agent_counter = 0
        self._agent_procs = {}  # agent_id -> Popen
        self._pending_agents = {}  # agent_id -> task string

    def _build_system_prompt(self) -> str:
        base_prompt = """You are a highly capable AI assistant powered by Ollama. 
You have access to a variety of tools to help you complete tasks.

CRITICAL RULES:
1. When you need to use a tool, you MUST use the EXACT format below.
2. For IMAGE tasks:
   - To CREATE/GENERATE a new image: use `generate_image` tool ONLY
   - To ANALYZE an existing image file: use `analyze_image` tool
3. Output ONLY the tool call. DO NOT explain what you are doing.
4. DO NOT repeat a tool call if you have already received a result for it.
5. If you have the answer, just give it. Do not use tools unless necessary.
6. After a tool succeeds (e.g., image generated, file written), respond with a brief confirmation. DO NOT call additional tools.
7. DO NOT try to read binary files (images, audio) with read_file - they are not text.

<tool_call>
<tool_name>name_of_tool</tool_name>
<parameters>{"param": "value"}</parameters>
</tool_call>

You can call multiple tools in one response by repeating the block above."""
        
        tool_desc = registry.generate_system_prompt_snippet()
        
        # Add MCP tools if any
        if MCP_TOOLS:
            tool_desc += "\nMCP Tools:\n"
            for name, info in MCP_TOOLS.items():
                tool_desc += f"- {name}: {info['description']}\n"
        
        return f"{base_prompt}\n\n{tool_desc}"

    def reset_messages(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def _save_checkpoint(self, prompt_text: str = ""):
        """Save current conversation state as a checkpoint."""
        import copy
        self.checkpoints.append((copy.deepcopy(self.messages), self.current_model, prompt_text))
        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints.pop(0)

    def _restore_checkpoint(self):
        """Show checkpoint list and let user pick which to restore."""
        if not self.checkpoints:
            print_status("No checkpoints to restore.")
            return

        # Build list with stored prompt text
        console.print("\n[bold cyan]Checkpoints (restore to before this prompt):[/bold cyan]")
        for i, (msgs, model, prompt_text) in enumerate(self.checkpoints):
            preview = prompt_text[:60] if prompt_text else "(start of conversation)"
            if len(prompt_text) > 60:
                preview += "..."
            console.print(f"  [dim]{i + 1}.[/dim] {preview}")
        console.print("")

        try:
            from prompt_toolkit import prompt as pt_prompt
            choice = pt_prompt("Restore checkpoint: ").strip()
        except (KeyboardInterrupt, EOFError):
            choice = ""

        if not choice:
            print_status("Cancelled.")
            return

        if not choice.lstrip('-').isdigit() or not (1 <= int(choice) <= len(self.checkpoints)):
            print_error(f"Invalid selection: {choice}. Enter a number between 1 and {len(self.checkpoints)}.")
            return

        idx = int(choice) - 1
        msgs, model, prompt_text = self.checkpoints[idx]
        # Discard this and all later checkpoints
        self.checkpoints = self.checkpoints[:idx]
        self.messages = msgs
        self.current_model = model
        print_status(f"Restored to before: [dim]{prompt_text[:60]}[/dim]")
        print_status(f"Model: [bold green]{model}[/bold green] ({len(self.checkpoints)} checkpoints remaining)")

    def _save_and_exit(self):
        """Auto-save session and exit with resume hint."""
        # Only save if there are user messages beyond system prompt
        has_content = any(m.get("role") == "user" for m in self.messages)
        if has_content:
            sid = save_session(
                self.messages,
                session_id=self.session_id,
                model=self.current_model,
                auto_model=self.auto_model,
                checkpoints=[(m, mdl, txt) for m, mdl, txt in self.checkpoints],
            )
            print_status(f"Session saved. Resume with:")
            console.print(f"  [bold green]ollama-cli --session {sid}[/bold green]\n")
        else:
            print_status("Goodbye!")
        sys.exit(0)

    def _resume_session(self, session_id: str):
        """Restore state from a saved session."""
        state = load_session(session_id)
        if not state:
            print_error(f"Session '{session_id}' not found.")
            return False
        self.session_id = session_id
        self.messages = state.get("messages", [])
        model = state.get("model", "")
        if model:
            self.current_model = model
        self.auto_model = state.get("auto_model", True)
        # Restore checkpoints (stored as lists, convert back to tuples)
        raw_cp = state.get("checkpoints", [])
        self.checkpoints = [(cp[0], cp[1], cp[2]) for cp in raw_cp if len(cp) >= 3]
        print_status(f"Resumed session [bold]{session_id}[/bold]")
        print_status(f"Model: [bold green]{self.current_model}[/bold green] | {len(self.messages)} messages | {len(self.checkpoints)} checkpoints")
        return True

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (~4 chars per token)."""
        return len(text) // 4

    def _show_context(self):
        """Show context window usage."""
        ctx_len = self.client.get_context_length(self.current_model)

        # Calculate token usage by category
        sys_tokens = 0
        user_tokens = 0
        assistant_tokens = 0
        tool_tokens = 0

        for msg in self.messages:
            content = msg.get("content", "")
            tokens = self._estimate_tokens(content)
            role = msg.get("role", "")
            if role == "system":
                sys_tokens += tokens
            elif role == "user":
                # Tool results are injected as user messages with "Tool Result" prefix
                if content.startswith("Tool Result"):
                    tool_tokens += tokens
                else:
                    user_tokens += tokens
            elif role == "assistant":
                assistant_tokens += tokens

        total_used = sys_tokens + user_tokens + assistant_tokens + tool_tokens
        free = max(0, ctx_len - total_used)
        pct_used = (total_used / ctx_len * 100) if ctx_len else 0

        # Visual bar — each segment colored by category proportion
        bar_width = 20
        # Calculate proportional segments
        segments = [
            ("green", sys_tokens),
            ("blue", user_tokens),
            ("magenta", assistant_tokens),
            ("yellow", tool_tokens),
        ]
        bar_chars = ""
        cells_placed = 0
        for color, tokens in segments:
            n_cells = round(tokens / ctx_len * bar_width) if ctx_len else 0
            for _ in range(n_cells):
                if cells_placed < bar_width:
                    bar_chars += f"[bold {color}]⛁[/bold {color}] "
                    cells_placed += 1
        # Overflow cells (over context limit)
        while cells_placed < min(filled_total := round(total_used / ctx_len * bar_width) if ctx_len else 0, bar_width):
            bar_chars += "[bold red]⛁[/bold red] "
            cells_placed += 1
        # Free space cells
        while cells_placed < bar_width:
            bar_chars += "[dim]⛶[/dim] "
            cells_placed += 1

        def _fmt(n):
            if n >= 1000:
                return f"{n / 1000:.1f}k"
            return str(n)

        pct_str = f"{pct_used:.0f}%"
        if pct_used > 100:
            pct_str = f"[bold red]{pct_used:.0f}% OVERFLOW[/bold red]"

        console.print(f"\n[bold cyan]Context Usage[/bold cyan]")
        console.print(f"  {bar_chars}  {self.current_model}")
        console.print(f"  {'  ' * bar_width}   {_fmt(total_used)}/{_fmt(ctx_len)} tokens ({pct_str})")
        console.print(f"")
        console.print(f"  [bold]Estimated usage by category[/bold]")
        console.print(f"  [green]⛁[/green] System prompt:  {_fmt(sys_tokens)} tokens")
        console.print(f"  [blue]⛁[/blue] User messages:  {_fmt(user_tokens)} tokens")
        console.print(f"  [magenta]⛁[/magenta] Assistant:      {_fmt(assistant_tokens)} tokens")
        console.print(f"  [yellow]⛁[/yellow] Tool results:   {_fmt(tool_tokens)} tokens")
        console.print(f"  [dim]⛶[/dim] Free space:     {_fmt(free)} tokens ({max(0, 100 - pct_used):.0f}%)")
        console.print("")

    def _next_agent_id(self) -> str:
        self._agent_counter += 1
        return f"agent-{self._agent_counter:03d}"

    def _handle_agent_command(self, parts):
        """Handle /agent subcommands."""
        if len(parts) < 2:
            self._agent_help()
            return

        subcmd = parts[1].lower()
        _subcommands = {'status', 'peek', 'stop', 'forget', 'help'}

        if subcmd == 'status':
            self._agent_status()
        elif subcmd == 'peek':
            if len(parts) < 3:
                agents = self.mailbox.list_agents()
                if not agents:
                    print_error("No agents to peek.")
                    return
                agent_id = sorted(agents)[-1]
            else:
                agent_id = parts[2]
            self._agent_peek(agent_id)
        elif subcmd == 'stop':
            if len(parts) < 3:
                agents = self.mailbox.list_agents()
                running = [a for a in agents if self.mailbox.get_status(a) == "running"]
                if not running:
                    print_error("No running agents to stop.")
                    return
                agent_id = running[-1]
            else:
                agent_id = parts[2]
            self._agent_stop(agent_id)
        elif subcmd == 'forget':
            if len(parts) < 3:
                print_error("Usage: /agent forget <agent_id>")
                return
            self._agent_forget(parts[2])
        elif subcmd == 'help' and len(parts) == 2:
            self._agent_help()
        else:
            # Everything after /agent is the task
            task = " ".join(parts[1:])
            self._agent_spawn(task)

    def _agent_help(self):
        console.print("""
[bold cyan]Agent Commands:[/bold cyan]
  /agent <task>       - Spawn a background subagent for a task
  /agent status       - Show all agents and their status
  /agent peek [id]    - Show full execution trace of an agent
  /agent stop [id]    - Stop a running agent and get partial result
  /agent forget <id>  - Remove agent's summary from context
  /agent help         - Show this help

  Agents run in the background. You'll be notified when they finish.
  Complex tasks are auto-split into parallel subtasks with dependencies.
""")

    def _agent_spawn(self, task: str):
        """Spawn a subagent for the given task."""
        agent_id = self._next_agent_id()
        model = self.current_model

        # Pick relevant tools based on task keywords
        tools = list(registry.tools.keys())

        print_status(f"Spawning [bold blue]{agent_id}[/bold blue] on [bold green]{model}[/bold green]...")
        print_status(f"Task: [dim]{task[:80]}[/dim]")

        proc = spawn_agent(
            task=task,
            model=model,
            tools=tools,
            agent_id=agent_id,
            mailbox_dir=self.mailbox.base_dir,
            ollama_url=self.config.get("ollama_url", "http://localhost:11434"),
        )
        self._agent_procs[agent_id] = proc
        self._pending_agents[agent_id] = task

        print_status(f"Agent running in background. Use [bold]/agent status[/bold] to check.")

    def _check_completed_agents(self):
        """Check for agents that finished since last prompt. Load their summaries."""
        if not self._pending_agents:
            return
        completed = []
        for agent_id, task in list(self._pending_agents.items()):
            status = poll_agent(agent_id, self.mailbox)
            if status in ("success", "done", "error"):
                completed.append((agent_id, task, status))

        for agent_id, task, status in completed:
            del self._pending_agents[agent_id]
            summary = self.mailbox.read_summary(agent_id)
            if summary:
                console.print(f"\n[bold cyan]✓ {agent_id} finished:[/bold cyan] {task[:50]}")
                console.print(f"  {summary[:200]}")
                if len(summary) > 200:
                    console.print(f"  [dim]... /agent peek {agent_id} for full trace[/dim]")
                console.print("")
                self.messages.append({
                    "role": "user",
                    "content": f"[Subagent {agent_id} completed task: {task}]\n\nResult: {summary}"
                })
            elif status == "error":
                print_error(f"Agent {agent_id} failed. Use /agent peek {agent_id} for details.")

    def _try_delegate(self, user_input: str) -> bool:
        """Ask the planner if this task should be delegated to subagents.

        Returns True if delegation happened (caller should skip normal chat).
        Returns False if the task should be handled directly.
        """
        import time

        print_status(f"Planning ([dim]{self.current_model}[/dim])...")
        subtasks = plan_subtasks(self.client, self.current_model, user_input)

        if not subtasks:
            return False

        # Compute execution waves based on dependencies
        waves = get_execution_waves(subtasks)
        has_deps = any(st.get("depends_on") for st in subtasks)

        console.print(f"\n[bold cyan]Breaking into {len(subtasks)} subtasks"
                       f"{' (with dependencies)' if has_deps else ''}:[/bold cyan]")
        for i, st in enumerate(subtasks):
            deps = st.get("depends_on", [])
            dep_str = f" [dim](after #{', #'.join(str(d) for d in deps)})[/dim]" if deps else ""
            console.print(f"  [dim]#{i}[/dim] {st['task'][:60]}{dep_str}")
        console.print("")

        # Execute wave by wave
        # Map subtask index → agent_id and results
        idx_to_agent = {}
        idx_to_summary = {}
        all_agent_ids = []  # (agent_id, task) for final collection

        try:
            for wave_num, wave_indices in enumerate(waves):
                if len(waves) > 1:
                    print_status(f"Wave {wave_num + 1}/{len(waves)}...")

                # Spawn all agents in this wave
                wave_agents = []
                for idx in wave_indices:
                    agent_id = self._next_agent_id()
                    st = subtasks[idx]
                    task = st["task"]
                    tools = st.get("tools", list(registry.tools.keys()))

                    # Inject dependency results as context
                    dep_context = ""
                    for dep_idx in st.get("depends_on", []):
                        dep_summary = idx_to_summary.get(dep_idx, "")
                        if dep_summary:
                            dep_task = subtasks[dep_idx]["task"]
                            dep_context += f"Result from prior subtask ({dep_task}): {dep_summary}\n"

                    proc = spawn_agent(
                        task=task,
                        model=self.current_model,
                        tools=tools,
                        agent_id=agent_id,
                        mailbox_dir=self.mailbox.base_dir,
                        ollama_url=self.config.get("ollama_url", "http://localhost:11434"),
                        context=dep_context,
                    )
                    self._agent_procs[agent_id] = proc
                    idx_to_agent[idx] = agent_id
                    wave_agents.append((idx, agent_id, task))
                    all_agent_ids.append((agent_id, task))

                # Wait for this wave to finish
                pending = set(aid for _, aid, _ in wave_agents)
                while pending:
                    for idx, agent_id, task in wave_agents:
                        if agent_id not in pending:
                            continue
                        status = poll_agent(agent_id, self.mailbox)
                        if status in ("success", "done", "error"):
                            pending.discard(agent_id)
                            summary = self.mailbox.read_summary(agent_id)
                            idx_to_summary[idx] = summary or ""
                            if summary:
                                console.print(f"  ✓ [bold]{agent_id}[/bold] — {task[:40]}")
                            else:
                                console.print(f"  ✗ [bold red]{agent_id}[/bold red] — failed")
                    if pending:
                        time.sleep(0.5)

        except KeyboardInterrupt:
            print_status("Detached. Agents still running in background.")
            for aid, task in all_agent_ids:
                if poll_agent(aid, self.mailbox) == "running":
                    self._pending_agents[aid] = task
            return True

        # Collect all summaries
        summaries = []
        for agent_id, task in all_agent_ids:
            summary = self.mailbox.read_summary(agent_id)
            if summary:
                summaries.append(f"[Subtask: {task}]\nResult: {summary}")

        if summaries:
            combined = "\n\n".join(summaries)

            # Context budget check
            summary_tokens = self._estimate_tokens(combined)
            ctx_len = self.client.get_context_length(self.current_model)
            current_tokens = sum(self._estimate_tokens(m.get("content", "")) for m in self.messages)
            if current_tokens + summary_tokens > ctx_len * 0.8:
                console.print(f"[bold yellow]Warning:[/bold yellow] Agent summaries use ~{summary_tokens} tokens. "
                              f"Context is {int((current_tokens + summary_tokens) / ctx_len * 100)}% full.")

            self.messages.append({
                "role": "user",
                "content": (
                    f"Subagents completed their research. Here are the results:\n\n"
                    f"{combined}\n\n"
                    f"[SYSTEM]: Using ONLY the subagent results above, provide a clear, "
                    f"complete answer to the user's original question: \"{user_input}\"\n"
                    f"DO NOT call any tools. DO NOT use placeholders. Just synthesize the data."
                )
            })
            console.print("")
            self.process_chat_cycle()
        else:
            print_error("All subagents failed to produce results.")

        return True

    def _agent_status(self):
        """Show status of all agents in this session."""
        agents = self.mailbox.list_agents()
        if not agents:
            print_status("No agents in this session.")
            return

        console.print(f"\n[bold cyan]Agents ({len(agents)}):[/bold cyan]")
        for agent_id in sorted(agents):
            status = self.mailbox.get_status(agent_id)
            steps = self.mailbox.read_steps(agent_id)

            # Get task from init step
            task = ""
            for s in steps:
                if s.get("step") == "init":
                    task = s.get("task", "")[:50]
                    break

            # Color by status
            if status in ("success", "done"):
                status_str = f"[green]{status}[/green]"
            elif status == "running":
                status_str = f"[yellow]{status}[/yellow]"
            elif status == "error":
                status_str = f"[red]{status}[/red]"
            else:
                status_str = f"[dim]{status}[/dim]"

            step_count = len([s for s in steps if s.get("step") in ("think", "act", "observe")])
            console.print(f"  [dim]{agent_id}[/dim] {status_str} ({step_count} steps) — {task}")
        console.print("")

    def _agent_peek(self, agent_id: str):
        """Show full execution trace (NOT loaded into orchestrator context)."""
        steps = self.mailbox.read_steps(agent_id)
        if not steps:
            print_error(f"Agent '{agent_id}' not found.")
            return

        console.print(f"\n[bold cyan]Trace for {agent_id}:[/bold cyan]")
        for step in steps:
            step_type = step.get("step", "?")

            if step_type == "init":
                console.print(f"  [bold]INIT[/bold] task=[dim]{step.get('task', '')}[/dim] model=[dim]{step.get('model', '')}[/dim]")
            elif step_type == "think":
                content = step.get("content", "")[:200]
                console.print(f"  [cyan]THINK[/cyan] {content}")
            elif step_type == "act":
                tool = step.get("tool", "?")
                params = json.dumps(step.get("params", {}))[:100]
                console.print(f"  [yellow]ACT[/yellow]   {tool}({params})")
            elif step_type == "observe":
                content = step.get("content", "")[:200]
                console.print(f"  [blue]OBS[/blue]   {content}")
            elif step_type == "done":
                summary = step.get("summary", "")[:200]
                status = step.get("status", "")
                console.print(f"  [green]DONE[/green]  [{status}] {summary}")
            elif step_type == "error":
                content = step.get("content", "")[:200]
                console.print(f"  [red]ERROR[/red] {content}")
            elif step_type == "signal":
                console.print(f"  [bold red]SIGNAL[/bold red] {step.get('signal', '')}")
            else:
                console.print(f"  [dim]{step_type}[/dim] {json.dumps(step)[:100]}")
        console.print("")

    def _agent_stop(self, agent_id: str):
        """Stop a running agent and load its summary into context."""
        status = self.mailbox.get_status(agent_id)
        if status not in ("running", "unknown"):
            print_status(f"Agent {agent_id} is not running (status: {status}).")
            # If it already finished, offer to load the summary
            summary = self.mailbox.read_summary(agent_id)
            if summary:
                console.print(f"[bold cyan]Summary from {agent_id}:[/bold cyan]")
                console.print(summary)
                console.print("")
            return

        print_status(f"Stopping [bold]{agent_id}[/bold]...")
        summary = stop_agent(agent_id, self.mailbox, timeout=15)
        if summary:
            console.print(f"[bold cyan]Summary from {agent_id}:[/bold cyan]")
            console.print(summary)
            console.print("")
            # Load into orchestrator context
            self.messages.append({
                "role": "user",
                "content": f"[Subagent {agent_id} was stopped]\n\nPartial result: {summary}"
            })
        else:
            print_error(f"Agent {agent_id} did not produce a summary within timeout.")

    def _agent_forget(self, agent_id: str):
        """Remove an agent's summary from orchestrator context and archive mailbox."""
        before = len(self.messages)
        self.messages = [
            m for m in self.messages
            if not (m.get("role") == "user" and
                    agent_id in m.get("content", "") and
                    ("Subagent" in m.get("content", "") or "subtask" in m.get("content", "").lower()))
        ]
        removed = before - len(self.messages)

        if removed > 0:
            freed = sum(self._estimate_tokens(m.get("content", ""))
                        for m in self.messages[len(self.messages):])  # already removed
            self.mailbox.cleanup(agent_id, archive=True)
            print_status(f"Forgot {agent_id}: removed {removed} message(s) from context. Mailbox archived.")
        else:
            print_status(f"No context entries found for {agent_id}. Use /agent status to see agents.")

    def handle_command(self, user_input: str) -> bool:
        """Handle slash commands."""
        if not user_input.startswith('/'):
            return False
            
        parts = user_input.split()
        cmd = parts[0].lower()
        
        try:
            if cmd == '/help':
                help_text = """
[bold cyan]Ollama CLI v3.0 Help[/bold cyan]

[bold]General Commands:[/bold]
  /help           - Show this help message
  /quit, /exit    - Save session and exit
  /clear          - Clear conversation history
  /context        - Show context window usage

[bold]Input:[/bold]
  Enter           - Submit prompt
  Alt+Enter / Ctrl+J - Insert newline (multiline input)
  ESC ESC         - Restore checkpoint
  Ctrl+C          - Cancel current input / stop response
  Ctrl+Z          - Suspend (resume with fg)

[bold]Model & Sessions:[/bold]
  /models         - List available models
  /model <name>   - Switch current model
  /sessions       - List saved sessions with previews
  /save [id]      - Save current session
  /load <id>      - Resume a saved session

[bold]Core Agent Tools:[/bold]
  [blue]Filesystem:[/blue] read_file, write_file, list_directory, grep_search, get_tree
  [blue]Code & Python:[/blue] run_python, replace_text, code_analyze_file, run_linter
  [blue]Web:[/blue] web_search, read_url
  [blue]Media:[/blue] generate_image, speak_text, get_comfy_status, analyze_image
  [blue]Knowledge:[/blue] kb_add, kb_search, remember_fact, recall_facts, clear_memory

[bold]Media Features (Usage Examples):[/bold]
  [green]Image Generation:[/green]
    "create an image of a sunset over mountains"
    "generate a 1024x768 image of a cat wearing a hat"
  
  [green]Image Analysis:[/green]
    "analyze image.jpg and describe what you see"
    "what's in /path/to/photo.png?"
  
  [green]Text-to-Speech:[/green]
    "speak: Hello, this is a test"
    "read this text aloud: [your text]"

[bold]Agents:[/bold]
  /agent <task>       - Spawn a subagent for a task
  /agent status       - Show all agents and their status
  /agent peek [id]    - Show full execution trace
  /agent stop [id]    - Stop a running agent
  /agent forget <id>  - Remove agent results from context

[bold]Integrations:[/bold]
  /mcp connect <name> <cmd> [args] - Connect MCP server
  /lsp start <lang> [path]         - Start LSP server
  /notify setup <topic>            - Setup ntfy.sh remote control
  
  [green]Notifications:[/green]
    After setup, send commands via: curl -d "your prompt" ntfy.sh/your-topic

[bold]Configuration:[/bold]
  /config ollama <url>       - Set Ollama server URL
  /config comfy <url>        - Set ComfyUI server URL
  /config comfy_path <path>  - Set ComfyUI install directory
  /config comfy_output <path> - Set ComfyUI output folder path
  /config piper_path <path>  - Set path to Piper binary
  /config piper_model <path> - Set path to Piper voice model
"""
                console.print(help_text)
            elif cmd == '/config':
                if len(parts) == 1:
                    cfg = load_config()
                    config_display = f"""
[bold cyan]Current Configuration:[/bold cyan]
  [blue]Ollama URL:[/blue]  {cfg.get('ollama_url')}
  [blue]ComfyUI URL:[/blue] {cfg.get('comfy_url')}
  [blue]ComfyUI Path:[/blue] {cfg.get('comfy_path')}
  [blue]Comfy Output:[/blue] {cfg.get('comfy_output_path')}
  [blue]Image Model:[/blue] {cfg.get('image_model')}
  [blue]Vision Model:[/blue]{cfg.get('vision_model')}
  [blue]Piper Path:[/blue]  {cfg.get('piper', {}).get('path')}
  [blue]Piper Model:[/blue] {cfg.get('piper', {}).get('model')}
  [blue]Default Model:[/blue] {cfg.get('default_model')}
"""
                    console.print(config_display)
                elif len(parts) > 2:
                    subcmd = parts[1].lower()
                    val = parts[2]
                    if subcmd == 'ollama':
                        update_config("ollama_url", val)
                        print_status(f"Ollama URL updated to: {val}")
                        self.client = OllamaClient(val)
                    elif subcmd == 'comfy':
                        update_config("comfy_url", val)
                        print_status(f"ComfyUI URL updated to: {val}")
                    elif subcmd == 'vision':
                        update_config("vision_model", val)
                        print_status(f"Vision model updated to: {val}")
                    elif subcmd == 'comfy_path':
                        update_config("comfy_path", val)
                        print_status(f"ComfyUI install path updated to: {val}")
                    elif subcmd == 'comfy_output':
                        update_config("comfy_output_path", val)
                        print_status(f"ComfyUI output path updated to: {val}")
                    elif subcmd == 'piper_path':
                        cfg = load_config()
                        cfg["piper"]["path"] = val
                        save_config(cfg)
                        print_status(f"Piper binary path updated to: {val}")
                    elif subcmd == 'piper_model':
                        cfg = load_config()
                        cfg["piper"]["model"] = val
                        save_config(cfg)
                        print_status(f"Piper model path updated to: {val}")
                    else:
                        print_error("Usage: /config <ollama|comfy|comfy_path|comfy_output|piper_path|piper_model> <value>")
                else:
                    print_error("Usage: /config <ollama|comfy|comfy_path|comfy_output|piper_path|piper_model> <value>")
            elif cmd in ('/quit', '/exit'):
                self._save_and_exit()
            elif cmd == '/context':
                self._show_context()
            elif cmd == '/clear':
                self.reset_messages()
                print_status("History cleared.")
            elif cmd == '/models':
                self.available_models = self.client.get_available_models()
                print_status(f"Available models: {', '.join(self.available_models)}")
            elif cmd == '/model':
                if len(parts) > 1:
                    if parts[1].lower() in ('auto', '0'):
                        self.auto_model = True
                        print_status("Switched to [bold green]auto[/bold green] model selection")
                    else:
                        if not self.available_models:
                            self.available_models = self.client.get_available_models()
                        if parts[1] in self.available_models:
                            self.current_model = parts[1]
                            self.auto_model = False
                            update_config("default_model", self.current_model)
                            print_status(f"Switched to model: {self.current_model}")
                        else:
                            print_error(f"Unknown model: {parts[1]}. Use /model to see available models.")
                else:
                    self.available_models = self.client.get_available_models()
                    if not self.available_models:
                        print_error("No models found. Make sure Ollama is running.")
                    else:
                        mode = "[bold green]auto[/bold green]" if self.auto_model else self.current_model
                        console.print(f"[bold cyan]Current model:[/bold cyan] {mode}\n")
                        auto_marker = " [bold green]*[/bold green]" if self.auto_model else ""
                        console.print(f"  [dim]0.[/dim] auto{auto_marker}")
                        for i, model in enumerate(self.available_models, 1):
                            marker = " [bold green]*[/bold green]" if not self.auto_model and model == self.current_model else ""
                            console.print(f"  [dim]{i}.[/dim] {model}{marker}")
                        console.print("")
                        try:
                            choice = self.repl.session.prompt(
                                HTML('<prompt>Select model (number or name, empty to cancel): </prompt>'),
                                style=self.repl.style
                            ).strip()
                        except (KeyboardInterrupt, EOFError):
                            choice = ""
                        if choice:
                            if choice == '0' or choice.lower() == 'auto':
                                self.auto_model = True
                                print_status("Switched to [bold green]auto[/bold green] model selection")
                            elif choice.lstrip('-').isdigit() and 1 <= int(choice) <= len(self.available_models):
                                self.current_model = self.available_models[int(choice) - 1]
                                self.auto_model = False
                                update_config("default_model", self.current_model)
                                print_status(f"Switched to model: {self.current_model}")
                            elif choice in self.available_models:
                                self.current_model = choice
                                self.auto_model = False
                                update_config("default_model", self.current_model)
                                print_status(f"Switched to model: {self.current_model}")
                            else:
                                print_error(f"Invalid selection: {choice}")
            elif cmd == '/sessions':
                sessions = list_sessions()
                if not sessions:
                    print_status("No saved sessions.")
                else:
                    console.print("\n[bold cyan]Saved Sessions:[/bold cyan]")
                    for s in sessions[:10]:
                        preview = get_session_preview(s)
                        console.print(f"  [dim]{s}[/dim] — {preview}")
                    console.print("")
            elif cmd == '/save':
                sid = save_session(
                    self.messages,
                    session_id=parts[1] if len(parts) > 1 else self.session_id,
                    model=self.current_model,
                    auto_model=self.auto_model,
                    checkpoints=[(m, mdl, txt) for m, mdl, txt in self.checkpoints],
                )
                print_status(f"Session saved: [bold]{sid}[/bold]")
                print_status(f"Resume with: [bold green]ollama-cli --session {sid}[/bold green]")
            elif cmd == '/load':
                if len(parts) > 1:
                    if self._resume_session(parts[1]):
                        pass  # success message printed by _resume_session
                    else:
                        print_error(f"Session '{parts[1]}' not found.")
                else:
                    print_error("Usage: /load <session_id>")
            elif cmd == '/mcp':
                if len(parts) > 3 and parts[1] == 'connect':
                    name = parts[2]
                    command = parts[3]
                    args = parts[4:] if len(parts) > 4 else None
                    result = connect_mcp_server(name, command, args)
                    print_status(result)
                    # Refresh system prompt with new tools
                    self.system_prompt = self._build_system_prompt()
                    self.reset_messages()
                else:
                    print_error("Usage: /mcp connect <name> <command> [args...]")
            elif cmd == '/lsp':
                if len(parts) > 2 and parts[1] == 'start':
                    lang = parts[2]
                    path = parts[3] if len(parts) > 3 else "."
                    result = start_lsp_server(lang, path)
                    print_status(result)
                else:
                    print_error("Usage: /lsp start <language> [path]")
            elif cmd == '/notify':
                if len(parts) > 2 and parts[1] == 'setup':
                    topic = parts[2]
                    self.config["ntfy"]["enabled"] = True
                    self.config["ntfy"]["topic"] = topic
                    save_config(self.config)
                    start_notify_thread(self.config, self.process_remote_message)
                    print_status(f"Notifications enabled for topic: {topic}")
                else:
                    print_error("Usage: /notify setup <topic>")
            elif cmd == '/agent':
                self._handle_agent_command(parts)
            else:
                print_error(f"Unknown command: {cmd}")
        except Exception as e:
            print_error(f"Command error: {e}")
            
        return True

    def process_remote_message(self, message: str) -> str:
        """Handle messages coming from ntfy.sh"""
        temp_messages = self.messages + [{"role": "user", "content": message}]
        try:
            response = ""
            for chunk in self.client.chat(temp_messages, self.current_model, stream=False):
                response += chunk
            return response
        except Exception as e:
            return f"Error: {e}"

    def run(self):
        console.print(f"[bold cyan]Ollama CLI v{VERSION}[/bold cyan]")

        # Initial model detection
        try:
            self.available_models = self.client.get_available_models()
            if not self.available_models:
                print_error("No models found. Make sure Ollama is running.")
            else:
                # Prefer config default, then llama3.2, then first available
                pref_model = self.config.get("default_model", "llama3.2")
                found_pref = False
                for am in self.available_models:
                    if pref_model in am:
                        self.current_model = am
                        found_pref = True
                        break

                if not found_pref:
                    self.current_model = self.available_models[0]

                print_status(f"Using model: [bold green]{self.current_model}[/bold green]")
        except Exception as e:
            print_error(f"Could not connect to Ollama: {e}")

        # Resume session if --session was provided, otherwise start fresh
        resumed = False
        if self._resume_id:
            resumed = self._resume_session(self._resume_id)

        if not resumed:
            self.reset_messages()
        
        # Start notification listener if enabled
        if self.config.get("ntfy", {}).get("enabled"):
            start_notify_thread(self.config, self.process_remote_message)
            print_status(f"Notification listener active on topic: {self.config['ntfy']['topic']}")

        while True:
            # Check if any background agents finished
            self._check_completed_agents()

            try:
                user_input = self.repl.get_input()
            except CheckpointRestore:
                self._restore_checkpoint()
                continue

            if not user_input:
                continue

            if self.handle_command(user_input):
                continue

            # Save checkpoint before processing the prompt
            self._save_checkpoint(user_input)

            self.messages.append({"role": "user", "content": user_input})

            # Auto-select best model for the task when in auto mode
            if self.auto_model and not self.current_model.startswith("llama3.2-vision"):
                task_model = self.client.select_best_model(user_input, self.available_models, self.current_model)
                if task_model != self.current_model:
                    print_status(f"Auto-switching to [bold green]{task_model}[/bold green]...")
                    self.current_model = task_model

            # Check if the task should be delegated to subagents
            if self._try_delegate(user_input):
                continue

            self.process_chat_cycle()

    def process_chat_cycle(self):
        """The main loop for chat and tool execution."""
        executed_tool_calls = set()
        for iteration in range(self.max_tool_iterations):
            full_response = ""
            try:
                print_status(f"Thinking ([dim]{self.current_model}[/dim])...")
                try:
                    with StreamingDisplay() as display:
                        for chunk in self.client.chat(self.messages, self.current_model):
                            display.update(chunk)
                            full_response += chunk
                except KeyboardInterrupt:
                    print_status("Response interrupted.")
                    if full_response:
                        self.messages.append({"role": "assistant", "content": full_response})
                    return

                self.messages.append({"role": "assistant", "content": full_response})
                
                # Check for tool calls
                tool_calls = self.parse_tool_calls(full_response)
                if not tool_calls:
                    break
                
                # Execute tool calls
                results = []
                for name, params in tool_calls:
                    # Prevent identical tool call loops across the entire cycle
                    call_key = f"{name}:{json.dumps(params, sort_keys=True)}"
                    if call_key in executed_tool_calls:
                        print_status(f"Skipping duplicate tool call: [dim]{name}[/dim]")
                        continue
                    
                    # Specific prevention for vision model loops: same tool + same file + already called once
                    if name == "analyze_image":
                        img_path = params.get("image_path")
                        if img_path:
                            vision_key = f"vision_path:{img_path}"
                            if vision_key in executed_tool_calls:
                                # We've already queried this image in this cycle.
                                # Check if the prompt is suspiciously similar to previous content
                                prompt = params.get("prompt", "").lower()
                                
                                # If prompt is long or contains parts of the previous messages, block it
                                is_suspicious = len(prompt) > 30
                                for msg in self.messages:
                                    if msg["role"] == "assistant" and prompt in msg["content"].lower():
                                        is_suspicious = True
                                        break
                                
                                if is_suspicious:
                                    print_status(f"Blocking suspicious vision loop for: [dim]{img_path}[/dim]")
                                    continue
                            executed_tool_calls.add(vision_key)

                    executed_tool_calls.add(call_key)
                    
                    result = self.execute_tool(name, params)
                    # Show result to user
                    preview = result[:200] + ("..." if len(result) > 200 else "")
                    print_status(f"Result: [dim]{preview}[/dim]")
                    results.append(f"Tool Result ({name}):\n{result}")
                
                if not results:
                    # If we had tool calls but they were all duplicates, we must stop
                    break
                    
                # Add all results to conversation
                feedback = "\n\n".join(results) + "\n\n[SYSTEM]: Tool execution is complete. Use the results above to provide your final response to the user. DO NOT call any tools again to process these results or to summarize them. If you have the information needed, answer the user now."
                self.messages.append({"role": "user", "content": feedback})
                
                # If we made it here, loop again to let the model process results
                
            except Exception as e:
                print_error(str(e))
                logger.exception("Chat cycle error")
                break

    def parse_tool_calls(self, text: str) -> List[tuple]:
        calls = []
        
        # 1. Try to find <tool_call> blocks (handles unclosed tags)
        tc_blocks = re.findall(r'<tool_call>(.*?)(?:</tool_call>|$)', text, re.DOTALL)
        if tc_blocks:
            for block in tc_blocks:
                if not block.strip(): continue
                name = None
                # Try to find tool_name tag (handles unclosed)
                name_match = re.search(r'<tool_name>(.*?)(?:</tool_name>|$)', block, re.DOTALL)
                if name_match:
                    name = name_match.group(1).strip()
                else:
                    # Fallback: tool name might be naked after <tool_call>
                    first_line = block.strip().split('\n')[0].strip()
                    if first_line in registry.tools:
                        name = first_line
                
                # Try to find parameters tag (handles unclosed)
                params_match = re.search(r'<parameters>(.*?)(?:</parameters>|$)', block, re.DOTALL)
                if name and params_match:
                    try:
                        params_str = params_match.group(1).strip()
                        # Auto-fix unclosed JSON if needed
                        if params_str.startswith('{') and not params_str.endswith('}'):
                            params_str += '}'
                        calls.append((name, json.loads(params_str)))
                    except:
                        # Last ditch: search for any JSON-like object in the parameters string
                        json_match = re.search(r'(\{.*\})', params_str, re.DOTALL)
                        if json_match:
                            try: calls.append((name, json.loads(json_match.group(1))))
                            except: pass
        
        # 2. If no <tool_call> blocks, try searching for tool-specific tags
        if not calls:
            for tool_name in registry.tools.keys():
                # Handle <tool_name ... /> or <tool_name ... >
                attr_pattern = fr'<{tool_name}\s+([^>]*?)[\s/]*>'
                attr_matches = re.findall(attr_pattern, text, re.DOTALL)
                for attr_str in attr_matches:
                    attrs = {}
                    # Find all pairs of key="value" or key='value'
                    kv_pairs = re.findall(r'(\w+)\s*=\s*(["\'])(.*?)\2', attr_str) # capture the quote type
                    for k, _, v in kv_pairs: # k, _, v to ignore the quote type
                        attrs[k] = v
                    if attrs:
                        calls.append((tool_name, attrs))
                
                # Handle <tool_name> {json} </tool_name>
                block_pattern = fr'<{tool_name}>(.*?)(?:</{tool_name}>|$)'
                block_matches = re.findall(block_pattern, text, re.DOTALL)
                for b in block_matches:
                    json_match = re.search(r'(\{.*\})', b, re.DOTALL)
                    if json_match:
                        try: calls.append((tool_name, json.loads(json_match.group(1))))
                        except: pass
        
        # 3. Final fallback: look for tool_name followed by JSON { ... }
        if not calls:
            for tool_name in registry.tools.keys():
                json_pattern = fr'{tool_name}\s*({{.*?}})'
                json_matches = re.findall(json_pattern, text, re.DOTALL)
                for json_str in json_matches:
                    try:
                        calls.append((tool_name, json.loads(json_str.strip())))
                    except:
                        pass
        
        # 4. Super-fallback: Line-by-line heuristic for "naked" calls (e.g. llama3.2:1b style)
        if not calls:
            lines = text.strip().split('\n')
            current_tool = None
            current_params = {}
            for line in lines:
                line = line.strip()
                if not line: continue
                
                # Look for a tool name at the start of the line (e.g. 'analyze_image path=...')
                # We use a word-boundary check to ensure it's the full tool name
                clean_line = line.lower().replace('tool_call', '').strip()
                match = re.search(r'^(\w+)', clean_line)
                t_name = match.group(1) if match else None
                
                if t_name in registry.tools:
                    # If we already had a tool, save it before starting new one
                    if current_tool and current_params:
                        calls.append((current_tool, current_params))
                        current_params = {}
                    current_tool = t_name
                    # Fall through to check for params on the same line
                
                # Look for key="value" or key='value' or key: value
                kv_match = re.findall(r'(\w+)\s*[=:]\s*(["\'])(.*?)\2', line)
                if kv_match and current_tool:
                    for k, _, v in kv_match:
                        # Map common misnamed parameters
                        if k == 'path' and current_tool == 'analyze_image': k = 'image_path'
                        current_params[k] = v
                elif current_tool:
                    # Try simplified key=value without quotes
                    kv_simple = re.findall(r'(\w+)\s*=\s*([^\s]+)', line)
                    for k, v in kv_simple:
                        if k == 'path' and current_tool == 'analyze_image': k = 'image_path'
                        current_params[k] = v
            
            if current_tool and current_params:
                calls.append((current_tool, current_params))

        return calls

    def execute_tool(self, name: str, params: Dict) -> str:
        """Execute a tool (standard or MCP)."""
        # Try MCP tools first
        if name in MCP_TOOLS:
            mcp_tool = MCP_TOOLS[name]
            print_status(f"Executing MCP tool [bold blue]{mcp_tool['name']}[/bold blue]...")
            return call_mcp_tool(mcp_tool["server"], mcp_tool["name"], params)
        
        # Try standard tools
        tool = registry.get_tool(name)
        if tool:
            print_status(f"Executing [bold blue]{name}[/bold blue]...")
            try:
                result = tool.execute(**params)
                # Notification for successful file/system ops
                if name == "write_file" and "Successfully" in str(result):
                    send_notification(self.config, "File Created", f"Path: {params.get('path')}")
                return result
            except Exception as e:
                return f"Error executing tool: {e}"
        
        return f"Unknown tool: {name}"

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ollama CLI")
    parser.add_argument("--session", type=str, default=None,
                        help="Resume a saved session by ID")
    args = parser.parse_args()

    # Ensure Ctrl+Z (SIGTSTP) uses the default handler so the process can be
    # suspended and resumed with fg.  Some libraries (e.g. prompt_toolkit) may
    # override this; restoring SIG_DFL lets the OS handle it natively.
    if hasattr(signal, "SIGTSTP"):
        signal.signal(signal.SIGTSTP, signal.SIG_DFL)

    cli = None
    try:
        cli = OllamaCLI(session_id=args.session)
        cli.run()
    except KeyboardInterrupt:
        if cli:
            try:
                cli._save_and_exit()
            except SystemExit:
                pass
        else:
            print_status("Interrupted by user. Exiting.")
            sys.exit(0)
    except Exception as e:
        print_error(f"Fatal error: {e}")
        logger.exception("Fatal error")
        sys.exit(1)

if __name__ == "__main__":
    main()
