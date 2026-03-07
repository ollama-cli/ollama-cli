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
from ollama_cli.core.sessions import save_session, load_session, list_sessions
from ollama_cli.ui.formatter import console, print_markdown, print_error, print_status, StreamingDisplay
from prompt_toolkit.formatted_text import HTML
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
        self.auto_model = True
        self.system_prompt = self._build_system_prompt()
        self.max_tool_iterations = 5

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

[bold]Integrations:[/bold]
  /mcp connect <name> <cmd> [args] - Connect MCP server
  /lsp start <lang> [path]         - Start LSP server
  /notify setup <topic>            - Setup ntfy.sh remote control
  
  [green]Notifications:[/green]
    After setup, send commands via: curl -d "your prompt" ntfy.sh/your-topic

[bold]Configuration:[/bold]
  /config ollama <url>       - Set Ollama server URL
  /config comfy <url>        - Set ComfyUI server URL
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
                        print_error("Usage: /config <ollama|comfy|comfy_output|piper_path|piper_model> <value>")
                else:
                    print_error("Usage: /config <ollama|comfy|comfy_output|piper_path|piper_model> <value>")
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
            
            # Auto-select best model for the task when in auto mode
            if self.auto_model and not self.current_model.startswith("llama3.2-vision"):
                task_model = self.client.select_best_model(user_input, self.available_models, self.current_model)
                if task_model != self.current_model:
                    print_status(f"Auto-switching to [bold green]{task_model}[/bold green]...")
                    self.current_model = task_model
                
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
                    print_status(f"Result: [dim]{result}[/dim]")
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
    # Ensure Ctrl+Z (SIGTSTP) uses the default handler so the process can be
    # suspended and resumed with fg.  Some libraries (e.g. prompt_toolkit) may
    # override this; restoring SIG_DFL lets the OS handle it natively.
    if hasattr(signal, "SIGTSTP"):
        signal.signal(signal.SIGTSTP, signal.SIG_DFL)

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
