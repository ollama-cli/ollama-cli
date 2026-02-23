"""
Ollama CLI v3.0
License: MIT
Author: Gemini CLI
"""

import sys
import os
import json
import re
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
from ollama_cli.core.sessions import save_session, load_session, list_sessions
from ollama_cli.ui.formatter import console, print_markdown, print_error, print_status, StreamingDisplay
from ollama_cli.ui.repl import REPL
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
    def __init__(self):
        self.config = load_config()
        self.client = OllamaClient(self.config.get("ollama_url"))
        self.repl = REPL()
        self.messages = []
        self.available_models = []
        self.current_model = self.config.get("default_model", "llama3.2")
        self.system_prompt = self._build_system_prompt()
        self.max_tool_iterations = 5

    def _build_system_prompt(self) -> str:
        base_prompt = """You are a highly capable AI assistant powered by Ollama. 
You have access to a variety of tools to help you complete tasks.

CRITICAL RULES:
1. When you need to use a tool, you MUST use the EXACT format below.
2. Output ONLY the tool call. DO NOT explain what you are doing.
3. DO NOT repeat a tool call if you have already received a result for it.
4. If you have the answer, just give it. Do not use tools unless necessary.

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
  /quit           - Exit the CLI
  /clear          - Clear conversation history

[bold]Model & Sessions:[/bold]
  /models         - List available models
  /model <name>   - Switch current model
  /sessions       - List saved sessions
  /save [name]    - Save current session
  /load <name>    - Load a saved session

[bold]Core Agent Tools:[/bold]
  [blue]Filesystem:[/blue] read_file, write_file, list_directory, grep_search, get_tree
  [blue]Code & Python:[/blue] run_python, replace_text, code_analyze_file, run_linter
  [blue]Web:[/blue] web_search, read_url
  [blue]Media:[/blue] generate_image, speak_text
  [blue]Knowledge:[/blue] kb_add, kb_search, remember_fact, recall_facts, clear_memory

[bold]Integrations:[/bold]
  /mcp connect <name> <cmd> [args] - Connect MCP server
  /lsp start <lang> [path]         - Start LSP server
  /notify setup <topic>            - Setup ntfy.sh remote control

[bold]Configuration:[/bold]
  /config ollama <url> - Set Ollama server URL (default: http://localhost:11434)
  /config comfy <url>  - Set ComfyUI server URL (default: http://127.0.0.1:8188)
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
  [blue]Image Model:[/blue] {cfg.get('image_model')}
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
                        print_error("Usage: /config <ollama|comfy|piper_path|piper_model> <value>")
                else:
                    print_error("Usage: /config <ollama|comfy|piper_path|piper_model> <value>")
            elif cmd == '/quit':
                print_status("Goodbye!")
                sys.exit(0)
            elif cmd == '/clear':
                self.reset_messages()
                print_status("History cleared.")
            elif cmd == '/models':
                self.available_models = self.client.get_available_models()
                print_status(f"Available models: {', '.join(self.available_models)}")
            elif cmd == '/model':
                if len(parts) > 1:
                    self.current_model = parts[1]
                    update_config("default_model", self.current_model)
                    print_status(f"Switched to model: {self.current_model}")
                else:
                    print_status(f"Current model: {self.current_model}")
            elif cmd == '/sessions':
                sessions = list_sessions()
                print_status(f"Recent sessions: {', '.join(sessions[:10])}")
            elif cmd == '/save':
                name = parts[1] if len(parts) > 1 else None
                path = save_session(self.messages, name)
                print_status(f"Session saved to {path}")
            elif cmd == '/load':
                if len(parts) > 1:
                    messages = load_session(parts[1])
                    if messages:
                        self.messages = messages
                        print_status(f"Session '{parts[1]}' loaded.")
                    else:
                        print_error(f"Session '{parts[1]}' not found.")
                else:
                    print_error("Usage: /load <session_name>")
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
                if self.current_model not in self.available_models:
                    self.current_model = self.available_models[0]
                print_status(f"Using model: [bold green]{self.current_model}[/bold green]")
        except Exception as e:
            print_error(f"Could not connect to Ollama: {e}")

        self.reset_messages()
        
        # Start notification listener if enabled
        if self.config.get("ntfy", {}).get("enabled"):
            start_notify_thread(self.config, self.process_remote_message)
            print_status(f"Notification listener active on topic: {self.config['ntfy']['topic']}")

        while True:
            user_input = self.repl.get_input()
            
            if not user_input:
                continue
                
            if self.handle_command(user_input):
                continue
                
            self.messages.append({"role": "user", "content": user_input})
            self.process_chat_cycle()

    def process_chat_cycle(self):
        """The main loop for chat and tool execution."""
        executed_tool_calls = set()
        for iteration in range(self.max_tool_iterations):
            full_response = ""
            try:
                print_status(f"Thinking ([dim]{self.current_model}[/dim])...")
                with StreamingDisplay() as display:
                    for chunk in self.client.chat(self.messages, self.current_model):
                        display.update(chunk)
                        full_response += chunk
                
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
                    executed_tool_calls.add(call_key)
                    
                    result = self.execute_tool(name, params)
                    # Show result to user
                    print_status(f"Result: [dim]{result}[/dim]")
                    results.append(f"Tool Result ({name}):\n{result}")
                
                if not results:
                    # If we had tool calls but they were all duplicates, we must stop
                    break
                    
                # Add all results to conversation
                self.messages.append({"role": "user", "content": "\n\n".join(results) + "\n\nTool execution complete. If the task is finished, provide a final response to the user. Do not repeat the tool call."})
                
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
                # Try to find tool_name tag (handles unclosed)
                name_match = re.search(r'<tool_name>(.*?)(?:</tool_name>|$)', block, re.DOTALL)
                name = name_match.group(1).strip() if name_match else None
                
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
                attr_pattern = fr'<{tool_name}\s+(.*?)[\s/]*>'
                attr_matches = re.findall(attr_pattern, text, re.DOTALL)
                for attr_str in attr_matches:
                    attrs = {}
                    kv_pairs = re.findall(r'(\w+)\s*=\s*["\'](.*?)["\']', attr_str)
                    for k, v in kv_pairs:
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
    try:
        cli = OllamaCLI()
        cli.run()
    except KeyboardInterrupt:
        print_status("Interrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        print_error(f"Fatal error: {e}")
        logger.exception("Fatal error")
        sys.exit(1)

if __name__ == "__main__":
    main()
