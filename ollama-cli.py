#!/usr/bin/env python3
import requests
import json
import subprocess
import os
import re
import sys
import threading
import queue
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"  # Default model
AVAILABLE_MODELS = []  # Cache available models
CONFIG_FILE = os.path.expanduser("~/.ollama-cli-config.json")
SESSION_DIR = os.path.expanduser("~/.ollama-cli-sessions")
CURRENT_SESSION = None
SESSION_CONTEXT = {
    "created_files": [],
    "last_output": "",
    "context_files": []
}
MCP_SERVERS = {}  # Connected MCP servers
MCP_TOOLS = {}  # Tools from MCP servers
LSP_SERVERS = {}  # Language servers
KNOWLEDGE_BASE = {}  # Simple knowledge store
SUBAGENTS = []  # Running subagents
NTFY_TOPIC = None  # ntfy.sh topic for notifications

def load_config():
    """Load config file"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        "excluded_models": [],
        "git": {
            "remote_url": "",
            "username": "",
            "password": ""
        },
        "gitea": {
            "url": "",
            "token": ""
        },
        "auto_save_sessions": True,
        "mcp_servers": {},
        "ollama_url": "http://localhost:11434/api/chat",
        "ntfy": {
            "enabled": False,
            "topic": "",
            "server": "https://ntfy.sh"
        }
    }

def send_notification(title, message, priority="default"):
    """Send notification via ntfy.sh"""
    config = load_config()
    ntfy_config = config.get("ntfy", {})
    
    if not ntfy_config.get("enabled") or not ntfy_config.get("topic"):
        return
    
    try:
        server = ntfy_config.get("server", "https://ntfy.sh")
        topic = ntfy_config["topic"]
        url = f"{server}/{topic}"
        
        requests.post(url, 
            data=message.encode('utf-8'),
            headers={
                "Title": title,
                "Priority": priority
            },
            timeout=5
        )
    except:
        pass  # Silently fail

def ntfy_listener(messages_list):
    """Listen for ntfy notifications and process them"""
    config = load_config()
    ntfy_config = config.get("ntfy", {})
    
    if not ntfy_config.get("enabled") or not ntfy_config.get("topic"):
        return
    
    server = ntfy_config.get("server", "https://ntfy.sh")
    topic = ntfy_config["topic"]
    url = f"{server}/{topic}/json"
    
    try:
        response = requests.get(url, stream=True, timeout=None)
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if data.get("event") == "message":
                    title = data.get("title", "")
                    message = data.get("message", "")
                    
                    # Skip messages sent by this script to avoid feedback loop
                    if title in ["Response", "Ollama CLI", "File Created", "Command Complete", "Test"]:
                        continue
                    
                    # Process notification as a user request
                    print(f"\n[📱 Notification: {title}]")
                    print(f"> {message}\n")
                    
                    messages_list.append({"role": "user", "content": message})
                    
                    # Get response
                    selected_model = select_best_model(message, "initial")
                    response_text = call_ollama(messages_list, selected_model)
                    messages_list.append({"role": "assistant", "content": response_text})
                    
                    # Extract clean output - remove tool calls and metadata
                    clean_output = re.sub(r'<tool_call>.*?</tool_call>', '', response_text, flags=re.DOTALL)
                    clean_output = '\n'.join([
                        line for line in clean_output.split('\n') 
                        if line.strip() and not any(line.strip().startswith(p) for p in 
                            ['Result:', 'Output:', 'Exit code:', '[Executing', '[Result:', '[Using'])
                    ])
                    
                    # Send response back as notification
                    send_notification("Response", clean_output.strip()[:1000] if clean_output.strip() else response_text[:1000])
    except:
        pass

def save_config(config):
    """Save config file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def save_session(messages, session_name=None):
    """Save conversation session"""
    os.makedirs(SESSION_DIR, exist_ok=True)
    if not session_name:
        from datetime import datetime
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    session_file = os.path.join(SESSION_DIR, f"{session_name}.json")
    with open(session_file, 'w') as f:
        json.dump({"messages": messages, "timestamp": session_name}, f, indent=2)
    return session_file

def load_session(session_name):
    """Load conversation session"""
    session_file = os.path.join(SESSION_DIR, f"{session_name}.json")
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            data = json.load(f)
            return data.get("messages", [])
    return None

def list_sessions():
    """List available sessions"""
    if not os.path.exists(SESSION_DIR):
        return []
    sessions = [f.replace('.json', '') for f in os.listdir(SESSION_DIR) if f.endswith('.json')]
    return sorted(sessions, reverse=True)

def connect_mcp_server(name, command, args=None):
    """Connect to an MCP server via stdio"""
    try:
        import subprocess
        cmd = [command] + (args or [])
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "ollama-cli", "version": "2.0"}
            }
        }
        
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        
        # Read response
        response = process.stdout.readline()
        if response:
            data = json.loads(response)
            if "result" in data:
                MCP_SERVERS[name] = process
                
                # List tools
                tools_request = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list"
                }
                process.stdin.write(json.dumps(tools_request) + "\n")
                process.stdin.flush()
                
                tools_response = process.stdout.readline()
                if tools_response:
                    tools_data = json.loads(tools_response)
                    if "result" in tools_data and "tools" in tools_data["result"]:
                        for tool in tools_data["result"]["tools"]:
                            tool_name = f"mcp_{name}_{tool['name']}"
                            MCP_TOOLS[tool_name] = {
                                "server": name,
                                "name": tool["name"],
                                "description": tool.get("description", ""),
                                "schema": tool.get("inputSchema", {})
                            }
                
                return f"Connected to MCP server '{name}' with {len([t for t in MCP_TOOLS if t.startswith(f'mcp_{name}_')])} tools"
        
        return f"Failed to initialize MCP server '{name}'"
    except Exception as e:
        return f"Error connecting to MCP server: {str(e)}"

def call_mcp_tool(server_name, tool_name, arguments):
    """Call a tool on an MCP server"""
    try:
        if server_name not in MCP_SERVERS:
            return f"Error: MCP server '{server_name}' not connected"
        
        process = MCP_SERVERS[server_name]
        
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        
        response = process.stdout.readline()
        if response:
            data = json.loads(response)
            if "result" in data:
                return json.dumps(data["result"], indent=2)
            elif "error" in data:
                return f"MCP Error: {data['error']}"
        
        return "No response from MCP server"
    except Exception as e:
        return f"Error calling MCP tool: {str(e)}"

# === LSP SUPPORT ===
def start_lsp_server(language, root_path="."):
    """Start a language server"""
    lsp_commands = {
        "python": ["pyright-langserver", "--stdio"],
        "javascript": ["typescript-language-server", "--stdio"],
        "typescript": ["typescript-language-server", "--stdio"],
        "rust": ["rust-analyzer"],
        "go": ["gopls"],
    }
    
    if language not in lsp_commands:
        return f"No LSP server configured for {language}"
    
    try:
        process = subprocess.Popen(
            lsp_commands[language],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.expanduser(root_path)
        )
        
        # Initialize
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "processId": os.getpid(),
                "rootUri": f"file://{os.path.abspath(os.path.expanduser(root_path))}",
                "capabilities": {}
            }
        }
        
        msg = json.dumps(init_request)
        process.stdin.write(f"Content-Length: {len(msg)}\r\n\r\n{msg}")
        process.stdin.flush()
        
        LSP_SERVERS[language] = process
        return f"Started LSP server for {language}"
    except Exception as e:
        return f"Error starting LSP: {str(e)}"

def lsp_goto_definition(language, file_path, line, character):
    """Get definition location"""
    if language not in LSP_SERVERS:
        return "LSP server not started"
    
    try:
        process = LSP_SERVERS[language]
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "textDocument/definition",
            "params": {
                "textDocument": {"uri": f"file://{os.path.abspath(file_path)}"},
                "position": {"line": line, "character": character}
            }
        }
        
        msg = json.dumps(request)
        process.stdin.write(f"Content-Length: {len(msg)}\r\n\r\n{msg}")
        process.stdin.flush()
        
        # Read response (simplified)
        return "Definition lookup sent (response parsing not implemented)"
    except Exception as e:
        return f"LSP error: {str(e)}"

def lsp_find_references(language, file_path, line, character):
    """Find all references to symbol"""
    if language not in LSP_SERVERS:
        return "LSP server not started"
    
    try:
        process = LSP_SERVERS[language]
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "textDocument/references",
            "params": {
                "textDocument": {"uri": f"file://{os.path.abspath(file_path)}"},
                "position": {"line": line, "character": character},
                "context": {"includeDeclaration": True}
            }
        }
        
        msg = json.dumps(request)
        process.stdin.write(f"Content-Length: {len(msg)}\r\n\r\n{msg}")
        process.stdin.flush()
        
        return "References lookup sent"
    except Exception as e:
        return f"LSP error: {str(e)}"

# === WEB SEARCH (Enhanced) ===
def web_search_enhanced(query):
    """Enhanced web search with DuckDuckGo"""
    try:
        import urllib.parse
        encoded = urllib.parse.quote(query)
        
        # Use DuckDuckGo Instant Answer API
        api_url = f"https://api.duckduckgo.com/?q={encoded}&format=json"
        response = requests.get(api_url, timeout=10)
        data = response.json()
        
        results = []
        if data.get("AbstractText"):
            results.append(f"Summary: {data['AbstractText']}")
        
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append(f"- {topic['Text']}")
        
        return "\n".join(results) if results else "No results found"
    except Exception as e:
        return f"Search error: {str(e)}"

# === KNOWLEDGE BASE ===
def kb_add(key, content):
    """Add to knowledge base"""
    KNOWLEDGE_BASE[key] = {
        "content": content,
        "timestamp": subprocess.run("date", capture_output=True, text=True).stdout.strip()
    }
    return f"Added '{key}' to knowledge base"

def kb_search(query):
    """Search knowledge base (simple keyword match)"""
    results = []
    query_lower = query.lower()
    for key, data in KNOWLEDGE_BASE.items():
        if query_lower in key.lower() or query_lower in data["content"].lower():
            results.append(f"{key}: {data['content'][:200]}...")
    return "\n".join(results) if results else "No matches found"

def kb_get(key):
    """Get from knowledge base"""
    if key in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[key]["content"]
    return f"Key '{key}' not found"

# === SUBAGENTS ===
def spawn_subagent(task, model=None):
    """Spawn a subagent to handle a task"""
    try:
        agent_id = len(SUBAGENTS) + 1
        
        # Create a simple prompt for the subagent
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Complete the task concisely."},
            {"role": "user", "content": task}
        ]
        
        # Run in thread
        result_queue = queue.Queue()
        
        def run_agent():
            try:
                response = call_ollama(messages, model)
                result_queue.put({"success": True, "result": response})
            except Exception as e:
                result_queue.put({"success": False, "error": str(e)})
        
        thread = threading.Thread(target=run_agent, daemon=True)
        thread.start()
        
        agent = {
            "id": agent_id,
            "task": task,
            "thread": thread,
            "queue": result_queue,
            "status": "running"
        }
        
        SUBAGENTS.append(agent)
        return f"Spawned subagent #{agent_id} for task: {task[:50]}..."
    except Exception as e:
        return f"Error spawning subagent: {str(e)}"

def check_subagents():
    """Check status of subagents"""
    results = []
    for agent in SUBAGENTS:
        if not agent["queue"].empty():
            result = agent["queue"].get()
            agent["status"] = "completed"
            agent["result"] = result
            results.append(f"Agent #{agent['id']}: {result.get('result', result.get('error'))[:200]}")
        elif agent["thread"].is_alive():
            results.append(f"Agent #{agent['id']}: Still running...")
        else:
            results.append(f"Agent #{agent['id']}: Completed")
    
    return "\n".join(results) if results else "No active subagents"

# === ADVANCED CODE TOOLS ===
def code_search_symbol(symbol_name, path="."):
    """Search for symbol definitions"""
    try:
        # Simple grep-based search
        result = subprocess.run(
            f'grep -rn "def {symbol_name}\\|class {symbol_name}\\|function {symbol_name}" {path}',
            shell=True, capture_output=True, text=True
        )
        return result.stdout if result.stdout else f"Symbol '{symbol_name}' not found"
    except Exception as e:
        return f"Error: {str(e)}"

def code_analyze_file(file_path):
    """Analyze code structure and suggest improvements"""
    try:
        with open(os.path.expanduser(file_path), 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        functions = [l for l in lines if 'def ' in l]
        classes = [l for l in lines if 'class ' in l]
        
        # Analyze issues
        issues = []
        if len(lines) > 500:
            issues.append(f"Large file ({len(lines)} lines) - consider splitting into modules")
        if len(functions) > 30:
            issues.append(f"Many functions ({len(functions)}) - consider organizing into classes")
        
        bare_excepts = sum(1 for l in lines if l.strip() == 'except:')
        if bare_excepts > 0:
            issues.append(f"{bare_excepts} bare except clauses - should catch specific exceptions")
        
        todos = sum(1 for l in lines if 'TODO' in l or 'FIXME' in l)
        if todos > 0:
            issues.append(f"{todos} TODO/FIXME comments")
        
        long_funcs = []
        in_func = False
        func_lines = 0
        func_name = ""
        for line in lines:
            if line.strip().startswith('def '):
                if in_func and func_lines > 50:
                    long_funcs.append(f"{func_name} ({func_lines} lines)")
                in_func = True
                func_lines = 0
                func_name = line.strip().split('(')[0].replace('def ', '')
            elif in_func:
                func_lines += 1
        
        if long_funcs:
            issues.append(f"Long functions: {', '.join(long_funcs[:3])}")
        
        return f"""File: {file_path}
Lines: {len(lines)} | Functions: {len(functions)} | Classes: {len(classes)}

Issues found:
{chr(10).join(f'- {i}' for i in issues) if issues else '- No major issues detected'}

Suggestions:
- Add type hints for better code clarity
- Add docstrings to undocumented functions
- Consider using logging instead of print statements
- Run a linter (pylint/flake8) for detailed analysis"""
    except Exception as e:
        return f"Error: {str(e)}"

# === DOCUMENTATION GENERATION ===
def generate_readme(path="."):
    """Generate README.md from project structure"""
    try:
        project_name = os.path.basename(os.path.abspath(path))
        
        # Analyze project
        files = []
        for root, dirs, filenames in os.walk(path):
            # Skip common ignore dirs
            dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__', 'venv']]
            for f in filenames:
                if f.endswith(('.py', '.js', '.ts', '.go', '.rs')):
                    files.append(os.path.join(root, f))
        
        # Detect language
        extensions = [os.path.splitext(f)[1] for f in files]
        main_lang = max(set(extensions), key=extensions.count) if extensions else '.txt'
        lang_map = {'.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript', '.go': 'Go', '.rs': 'Rust'}
        language = lang_map.get(main_lang, 'Unknown')
        
        readme = f"""# {project_name}

## Description
A {language} project with {len(files)} source files.

## Project Structure
```
{chr(10).join([f'- {os.path.relpath(f, path)}' for f in files[:10]])}
{'...' if len(files) > 10 else ''}
```

## Installation
```bash
# Add installation instructions here
```

## Usage
```bash
# Add usage examples here
```

## License
MIT
"""
        
        readme_path = os.path.join(path, "README.md")
        with open(readme_path, 'w') as f:
            f.write(readme)
        
        return f"Generated README.md at {readme_path}"
    except Exception as e:
        return f"Error generating README: {str(e)}"

def generate_docstrings(file_path):
    """Generate docstrings for functions in a file"""
    try:
        with open(os.path.expanduser(file_path), 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Detect function definition
            if line.strip().startswith('def ') and ':' in line:
                # Check if next line is already a docstring
                if i + 1 < len(lines) and '"""' not in lines[i + 1]:
                    # Extract function name
                    func_name = line.strip().split('(')[0].replace('def ', '')
                    indent = len(line) - len(line.lstrip())
                    docstring = f'{" " * (indent + 4)}"""TODO: Document {func_name}"""\n'
                    new_lines.append(line)
                    new_lines.append(docstring)
                    i += 1
                    continue
            new_lines.append(line)
            i += 1
        
        # Write back
        with open(os.path.expanduser(file_path), 'w') as f:
            f.writelines(new_lines)
        
        return f"Added docstring placeholders to {file_path}"
    except Exception as e:
        return f"Error: {str(e)}"

# === CODE QUALITY TOOLS ===
def run_linter(file_path, linter="auto"):
    """Run linter on file"""
    try:
        file_path = os.path.expanduser(file_path)
        ext = os.path.splitext(file_path)[1]
        
        # Auto-detect linter
        if linter == "auto":
            linter_map = {
                '.py': 'pylint',
                '.js': 'eslint',
                '.ts': 'eslint',
                '.go': 'golint',
                '.rs': 'cargo clippy'
            }
            linter = linter_map.get(ext, 'pylint')
        
        # Run linter
        result = subprocess.run(
            f"{linter} {file_path}",
            shell=True, capture_output=True, text=True, timeout=30
        )
        
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error running linter: {str(e)}"

def format_code(file_path, formatter="auto"):
    """Format code file"""
    try:
        file_path = os.path.expanduser(file_path)
        ext = os.path.splitext(file_path)[1]
        
        # Auto-detect formatter
        if formatter == "auto":
            formatter_map = {
                '.py': 'black',
                '.js': 'prettier --write',
                '.ts': 'prettier --write',
                '.go': 'gofmt -w',
                '.rs': 'rustfmt'
            }
            formatter = formatter_map.get(ext, 'cat')
        
        # Run formatter
        result = subprocess.run(
            f"{formatter} {file_path}",
            shell=True, capture_output=True, text=True, timeout=30
        )
        
        return f"Formatted {file_path}\n{result.stdout + result.stderr}"
    except Exception as e:
        return f"Error formatting: {str(e)}"

def check_security(path="."):
    """Run security checks"""
    try:
        # Try bandit for Python
        result = subprocess.run(
            f"bandit -r {path} 2>&1 || echo 'Install: pip install bandit'",
            shell=True, capture_output=True, text=True, timeout=60
        )
        return result.stdout
    except Exception as e:
        return f"Error: {str(e)}"

# === PROJECT TEMPLATES ===
def create_project(name, template="python"):
    """Create new project from template"""
    try:
        project_path = os.path.expanduser(f"./{name}")
        os.makedirs(project_path, exist_ok=True)
        
        templates = {
            "python": {
                "main.py": "#!/usr/bin/env python3\n\ndef main():\n    print('Hello, World!')\n\nif __name__ == '__main__':\n    main()\n",
                "requirements.txt": "# Add dependencies here\n",
                "README.md": f"# {name}\n\nA Python project.\n",
                ".gitignore": "__pycache__/\n*.pyc\nvenv/\n.env\n"
            },
            "javascript": {
                "index.js": "console.log('Hello, World!');\n",
                "package.json": f'{{"name": "{name}", "version": "1.0.0", "main": "index.js"}}',
                "README.md": f"# {name}\n\nA JavaScript project.\n",
                ".gitignore": "node_modules/\n.env\n"
            },
            "flask": {
                "app.py": "from flask import Flask\n\napp = Flask(__name__)\n\n@app.route('/')\ndef hello():\n    return 'Hello, World!'\n\nif __name__ == '__main__':\n    app.run(debug=True)\n",
                "requirements.txt": "flask\n",
                "README.md": f"# {name}\n\nA Flask web application.\n",
                ".gitignore": "__pycache__/\nvenv/\n*.pyc\n"
            },
            "express": {
                "server.js": "const express = require('express');\nconst app = express();\n\napp.get('/', (req, res) => {\n  res.send('Hello, World!');\n});\n\napp.listen(3000, () => console.log('Server running on port 3000'));\n",
                "package.json": f'{{"name": "{name}", "version": "1.0.0", "main": "server.js", "dependencies": {{"express": "^4.18.0"}}}}',
                "README.md": f"# {name}\n\nAn Express.js application.\n",
                ".gitignore": "node_modules/\n.env\n"
            }
        }
        
        if template not in templates:
            return f"Unknown template: {template}. Available: {', '.join(templates.keys())}"
        
        # Create files
        for filename, content in templates[template].items():
            filepath = os.path.join(project_path, filename)
            with open(filepath, 'w') as f:
                f.write(content)
        
        # Initialize git
        subprocess.run("git init", shell=True, cwd=project_path, capture_output=True)
        
        return f"Created {template} project at {project_path}\nFiles: {', '.join(templates[template].keys())}"
    except Exception as e:
        return f"Error creating project: {str(e)}"

# Model selection based on task type
MODEL_PREFERENCES = {
    "code": ["qwen3-coder:30b", "mistral:latest", "llama3:latest", "deepseek-coder-v2:16b", "granite-code:latest"],
    "general": ["mistral:latest", "llama3:latest", "llama3.2:latest"],
    "fast": ["llama3.2:3b", "llama3.2:latest"],
}

def get_available_models():
    """Get list of models available in Ollama"""
    config = load_config()
    excluded = config.get("excluded_models", [])
    
    try:
        result = subprocess.run("ollama list", shell=True, capture_output=True, text=True, timeout=5)
        models = []
        for line in result.stdout.split('\n')[1:]:  # Skip header
            if line.strip():
                model_name = line.split()[0]
                # Skip base models, tiny models, and excluded models
                if ('base' not in model_name.lower() and 
                    ':1b' not in model_name.lower() and
                    model_name not in excluded):
                    models.append(model_name)
        return models
    except:
        return []

def select_best_model(user_input, task_type="initial"):
    """Select best model based on user input and task type"""
    if not AVAILABLE_MODELS:
        return MODEL
    
    # For tool execution, prefer fast general models
    if task_type == "tool_response":
        for model in MODEL_PREFERENCES["general"]:
            if any(model in m for m in AVAILABLE_MODELS):
                return next(m for m in AVAILABLE_MODELS if model in m)
    
    # Detect task type from user input
    code_keywords = ['code', 'script', 'program', 'function', 'debug', 'python', 'javascript', 'write a', 'create a']
    
    if any(keyword in user_input.lower() for keyword in code_keywords):
        # Prefer code models
        for model in MODEL_PREFERENCES["code"]:
            if any(model in m for m in AVAILABLE_MODELS):
                return next(m for m in AVAILABLE_MODELS if model in m)
    
    # Default to general models
    for model in MODEL_PREFERENCES["general"]:
        if any(model in m for m in AVAILABLE_MODELS):
            return next(m for m in AVAILABLE_MODELS if model in m)
    
    # Fallback to first available
    return AVAILABLE_MODELS[0] if AVAILABLE_MODELS else MODEL

def execute_bash(command):
    """Execute bash command and return output"""
    # Check for dangerous commands
    dangerous = ['rm -rf', 'dd if=', 'mkfs', '> /dev/', 'format', 'sudo rm', ':(){:|:&};:']
    if any(d in command.lower() for d in dangerous):
        return f"⚠️  DANGEROUS COMMAND BLOCKED: {command}\nThis command could cause data loss. If you really need to run it, do so manually."
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, 
                              cwd=os.getcwd(), timeout=30)
        output = result.stdout + result.stderr
        return f"Exit code: {result.returncode}\n{output}" if output else f"Exit code: {result.returncode} (no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        return f"Error: {str(e)}"

def read_file(path):
    """Read file contents"""
    try:
        with open(os.path.expanduser(path), 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(path, content):
    """Write content to file"""
    try:
        expanded_path = os.path.expanduser(path)
        
        # For Python files, try to validate and fix basic issues
        if path.endswith('.py'):
            # Try to compile to check for syntax errors
            try:
                compile(content, expanded_path, 'exec')
            except SyntaxError as e:
                return f"Error: Generated Python code has syntax error at line {e.lineno}: {e.msg}"
        
        os.makedirs(os.path.dirname(expanded_path), exist_ok=True)
        with open(expanded_path, 'w') as f:
            f.write(content)
        
        # Track created file
        SESSION_CONTEXT["created_files"].append(expanded_path)
        
        return f"Successfully written to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def edit_file(path, old_str, new_str):
    """Edit file by replacing old_str with new_str"""
    try:
        expanded_path = os.path.expanduser(path)
        with open(expanded_path, 'r') as f:
            content = f.read()
        
        if old_str not in content:
            return f"Error: Could not find the specified text in {path}"
        
        new_content = content.replace(old_str, new_str, 1)
        
        # Validate Python syntax if applicable
        if path.endswith('.py'):
            try:
                compile(new_content, expanded_path, 'exec')
            except SyntaxError as e:
                return f"Error: Edited code has syntax error at line {e.lineno}: {e.msg}"
        
        with open(expanded_path, 'w') as f:
            f.write(new_content)
        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error editing file: {str(e)}"

def append_file(path, content):
    """Append content to file"""
    try:
        expanded_path = os.path.expanduser(path)
        with open(expanded_path, 'a') as f:
            f.write(content)
        return f"Successfully appended to {path}"
    except Exception as e:
        return f"Error appending to file: {str(e)}"

def search_files(pattern, path="."):
    """Search for pattern in files"""
    try:
        result = subprocess.run(
            f'grep -r "{pattern}" {path}',
            shell=True, capture_output=True, text=True, timeout=10
        )
        return result.stdout if result.stdout else "No matches found"
    except Exception as e:
        return f"Error searching: {str(e)}"

def git_operation(operation, message=None, **kwargs):
    """Perform git operations"""
    config = load_config()
    git_config = config.get("git", {})
    gitea_config = config.get("gitea", {})
    
    try:
        if operation == "status":
            result = subprocess.run("git status", shell=True, capture_output=True, text=True)
        elif operation == "diff":
            result = subprocess.run("git diff", shell=True, capture_output=True, text=True)
        elif operation == "commit":
            msg = message or kwargs.get("message", "Auto commit")
            result = subprocess.run(f'git add -A && git commit -m "{msg}"', 
                                  shell=True, capture_output=True, text=True)
        elif operation == "push":
            remote = git_config.get("remote_url", "origin")
            username = git_config.get("username", "")
            password = git_config.get("password", "")
            
            if username and password:
                # Inject credentials into URL
                result = subprocess.run(f"git push {remote}", 
                                      shell=True, capture_output=True, text=True)
            else:
                result = subprocess.run("git push", shell=True, capture_output=True, text=True)
        elif operation == "pull":
            result = subprocess.run("git pull", shell=True, capture_output=True, text=True)
        elif operation == "list":
            # Check if Gitea is configured
            if gitea_config.get("url") and gitea_config.get("token"):
                # Query Gitea API
                gitea_url = gitea_config["url"].rstrip("/")
                token = gitea_config["token"]
                
                try:
                    response = requests.get(
                        f"{gitea_url}/api/v1/user/repos",
                        headers={"Authorization": f"token {token}"},
                        timeout=10,
                        verify=False  # Skip SSL verification for localhost
                    )
                    
                    if response.status_code == 200:
                        repos = response.json()
                        result_text = "Repositories on Gitea server:\n"
                        for repo in repos:
                            result_text += f"- {repo['full_name']} ({repo['clone_url']})\n"
                        return result_text
                    else:
                        return f"Gitea API error: {response.status_code}"
                except Exception as e:
                    return f"Error querying Gitea: {str(e)}"
            else:
                # Fallback to local search
                result = subprocess.run("find . -maxdepth 2 -name '.git' -type d | sed 's|/.git||'", 
                                      shell=True, capture_output=True, text=True)
        else:
            return f"Unknown git operation: {operation}"
        
        return result.stdout + result.stderr
    except Exception as e:
        return f"Git error: {str(e)}"

def install_package(package, manager="pip"):
    """Install package using pip or npm"""
    try:
        if manager == "pip":
            result = subprocess.run(f"pip install {package}", 
                                  shell=True, capture_output=True, text=True, timeout=60)
        elif manager == "npm":
            result = subprocess.run(f"npm install {package}", 
                                  shell=True, capture_output=True, text=True, timeout=60)
        else:
            return f"Unknown package manager: {manager}"
        
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error installing package: {str(e)}"

def web_search(query):
    """Simple web search using curl"""
    try:
        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        result = subprocess.run(
            f'curl -s "https://html.duckduckgo.com/html/?q={encoded_query}"',
            shell=True, capture_output=True, text=True, timeout=10
        )
        # Extract snippets (very basic)
        content = result.stdout[:3000]
        return f"Search results for '{query}':\n{content}"
    except Exception as e:
        return f"Error searching: {str(e)}"

def list_directory(path="."):
    """List directory contents"""
    try:
        expanded = os.path.expanduser(path)
        files = os.listdir(expanded)
        return "\n".join(files) if files else "Directory is empty"
    except Exception as e:
        return f"Error listing directory: {str(e)}"

def get_file_info(path):
    """Get file metadata"""
    try:
        p = os.path.expanduser(path)
        stat = os.stat(p)
        import time
        return f"Path: {p}\nSize: {stat.st_size} bytes\nModified: {time.ctime(stat.st_mtime)}\nPermissions: {oct(stat.st_mode)}"
    except Exception as e:
        return f"Error getting file info: {str(e)}"

TOOLS = {
    "execute_bash": execute_bash,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "append_file": append_file,
    "web_search": web_search,
    "web_search_enhanced": web_search_enhanced,
    "list_directory": list_directory,
    "get_file_info": get_file_info,
    "search_files": search_files,
    "git_operation": git_operation,
    "install_package": install_package,
    "lsp_goto_definition": lsp_goto_definition,
    "lsp_find_references": lsp_find_references,
    "kb_add": kb_add,
    "kb_search": kb_search,
    "kb_get": kb_get,
    "spawn_subagent": spawn_subagent,
    "check_subagents": check_subagents,
    "code_search_symbol": code_search_symbol,
    "code_analyze_file": code_analyze_file,
    "generate_readme": generate_readme,
    "generate_docstrings": generate_docstrings,
    "run_linter": run_linter,
    "format_code": format_code,
    "check_security": check_security,
    "create_project": create_project,
}

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

WHEN TO USE TOOLS:
- User says "create a FILE/SCRIPT" → use write_file
- User says "edit/modify FILE" → use edit_file
- User says "add to FILE" → use append_file
- User asks to READ a file → use read_file
- User asks to RUN/EXECUTE a command → use execute_bash
- User asks to LIST files/directories → use list_directory
- User asks to LIST GIT PROJECTS/REPOS → use git_operation with operation="list"
- User asks to SEARCH in files → use search_files
- User asks about GIT (status, commit, push, pull) → use git_operation
- User asks to INSTALL package → use install_package
- User asks to SEARCH the web → use web_search_enhanced
- User asks about code definitions/references → use lsp_goto_definition, lsp_find_references
- User wants to save/retrieve knowledge → use kb_add, kb_search, kb_get
- User has complex multi-step task → use spawn_subagent
- User wants code analysis → use code_search_symbol, code_analyze_file
- User wants documentation → use generate_readme, generate_docstrings
- User wants code quality checks → use run_linter, format_code, check_security
- User wants to create new project → use create_project

WHEN NOT TO USE TOOLS (just answer normally):
- "how do I make FOOD" → answer with recipe
- "how to do X" → answer with instructions
- "what is X" → answer with explanation
- General questions about cooking, advice, information

Available tools:
- write_file, edit_file, append_file, read_file
- execute_bash, list_directory, get_file_info, search_files
- git_operation, install_package
- web_search_enhanced: Better web search with API
- lsp_goto_definition: Find symbol definition. Params: {"language": "python", "file_path": "file.py", "line": 0, "character": 0}
- lsp_find_references: Find all references. Params: {"language": "python", "file_path": "file.py", "line": 0, "character": 0}
- kb_add: Add to knowledge base. Params: {"key": "string", "content": "string"}
- kb_search: Search knowledge base. Params: {"query": "string"}
- kb_get: Get from knowledge base. Params: {"key": "string"}
- spawn_subagent: Spawn parallel task. Params: {"task": "string", "model": "optional"}
- check_subagents: Check subagent status
- code_search_symbol: Find symbol in code. Params: {"symbol_name": "string", "path": "."}
- code_analyze_file: Analyze code structure. Params: {"file_path": "string"}
- generate_readme: Generate README.md. Params: {"path": "."}
- generate_docstrings: Add docstrings to file. Params: {"file_path": "string"}
- run_linter: Run linter. Params: {"file_path": "string", "linter": "auto|pylint|eslint"}
- format_code: Format code. Params: {"file_path": "string", "formatter": "auto|black|prettier"}
- check_security: Security scan. Params: {"path": "."}
- create_project: Create project from template. Params: {"name": "string", "template": "python|javascript|flask|express"}

MCP Tools: {mcp_tools_list}

Tool call format:
<tool_call>
<tool_name>write_file</tool_name>
<parameters>{"path": "~/file.py", "content": "code here"}</parameters>
</tool_call>

Output ONLY the tool call when using a tool. Use \\n for newlines, proper indentation (4 spaces for Python)."""

def extract_code_blocks(text):
    """Extract code blocks from markdown"""
    pattern = r'```(\w+)?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return [(lang, code.strip()) for lang, code in matches]

def auto_save_code(response, user_input):
    """Automatically save code blocks if user asked to create a file"""
    create_keywords = ['create', 'write', 'make', 'generate', 'build']
    file_keywords = ['script', 'file', 'program', '.py', '.js', '.html', '.sh']
    
    should_save = any(k in user_input.lower() for k in create_keywords) and \
                  any(k in user_input.lower() for k in file_keywords)
    
    if not should_save:
        return None
    
    code_blocks = extract_code_blocks(response)
    if not code_blocks:
        return None
    
    # Get the largest code block (likely the main code)
    lang, code = max(code_blocks, key=lambda x: len(x[1]))
    
    if not code:
        return None
    
    # Determine filename from language or user input
    extensions = {
        'python': '.py',
        'javascript': '.js',
        'html': '.html',
        'bash': '.sh',
        'sh': '.sh',
    }
    
    # Try to extract filename from user input
    words = user_input.lower().split()
    filename = None
    for i, word in enumerate(words):
        if any(ext in word for ext in ['.py', '.js', '.html', '.sh']):
            filename = word
            break
        if word in ['called', 'named'] and i + 1 < len(words):
            filename = words[i + 1]
            break
    
    if not filename:
        # Generate filename based on language
        ext = extensions.get(lang.lower(), '.txt')
        if 'calculator' in user_input.lower():
            filename = f'calculator{ext}'
        elif 'script' in user_input.lower():
            filename = f'script{ext}'
        else:
            filename = f'output{ext}'
    
    if not filename:
        return None
        
    return filename, code

def parse_tool_calls(text):
    """Extract tool calls from model response"""
    calls = []
    
    # Try standard format first
    pattern = r'<tool_call>\s*<tool_name>(.*?)</tool_name>\s*<parameters>(.*?)</parameters>\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Try DeepSeek format: <｜tool▁call▁begin｜>function<｜tool▁sep｜>TOOL_NAME\n```json\n{...}\n```
    if not matches:
        ds_pattern = r'<｜tool▁call▁begin｜>function<｜tool▁sep｜>(\w+)\s*```json\s*(\{.*?\})\s*```'
        ds_matches = re.findall(ds_pattern, text, re.DOTALL)
        if ds_matches:
            matches = ds_matches
    
    for name, params in matches:
        params_str = params.strip()
        name = name.strip()
        
        # Try JSON first
        try:
            calls.append((name, json.loads(params_str)))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try ast.literal_eval (handles Python literals)
        try:
            import ast
            calls.append((name, ast.literal_eval(params_str)))
            continue
        except:
            pass
        
        # Try fixing common issues: literal newlines in JSON strings
        try:
            # Replace literal newlines with \n escape sequence
            fixed = params_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            calls.append((name, json.loads(fixed)))
            continue
        except:
            pass
        
        print(f"Warning: Failed to parse parameters for {name}")
        print(f"Raw params: {params_str[:100]}...")
    
    return calls

def call_ollama(messages, model=None):
    """Call Ollama API with streaming"""
    config = load_config()
    ollama_url = config.get("ollama_url", OLLAMA_URL)
    selected_model = model or MODEL
    try:
        response = requests.post(ollama_url, json={
            "model": selected_model,
            "messages": messages,
            "stream": True
        }, stream=True, timeout=60)
        response.raise_for_status()
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    print(content, end="", flush=True)
                    full_response += content
        print()  # Newline after streaming
        return full_response
    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to Ollama. Make sure it's running with 'ollama serve'")
        sys.exit(1)
    except Exception as e:
        return f"Error calling Ollama: {str(e)}"

def trim_messages(messages, max_messages=20):
    """Keep only recent messages to avoid context overflow"""
    if len(messages) > max_messages:
        # Keep system prompt + last (max_messages - 1) messages
        return [messages[0]] + messages[-(max_messages-1):]
    return messages

def main():
    global AVAILABLE_MODELS, CURRENT_SESSION, MODEL
    AVAILABLE_MODELS = get_available_models()
    
    print(f"Ollama CLI v2.0 - Default model: {MODEL}")
    
    config = load_config()
    ollama_url = config.get("ollama_url", OLLAMA_URL)
    print(f"Server: {ollama_url}")
    
    if AVAILABLE_MODELS:
        print(f"Available models: {', '.join(AVAILABLE_MODELS)}")
    else:
        print("Warning: Could not detect available models, using default")
    
    config = load_config()
    if config.get("excluded_models"):
        print(f"Excluded models: {', '.join(config['excluded_models'])}")
    
    print(f"Current directory: {os.getcwd()}")
    print(f"Home directory: {os.path.expanduser('~')}")
    print("\nCommands:")
    print("  /quit - Exit")
    print("  /clear - Clear history")
    print("  /save [name] - Save session")
    print("  /load <name> - Load session")
    print("  /sessions - List sessions")
    print("  /model <name> - Switch model")
    print("  /models - List models")
    print("  /exclude <model> - Exclude model")
    print("  /retry - Retry last request")
    print("  /run <file> - Execute file")
    print("  /config git <url> <user> <pass> - Configure git")
    print("  /config gitea <url> <token> - Configure Gitea server")
    print("  /export <file> - Export to markdown")
    print("  /mcp connect <name> <command> [args] - Connect MCP server")
    print("  /mcp list - List MCP servers and tools")
    print("  /mcp disconnect <name> - Disconnect MCP server")
    print("  /context <file> - Add file to context")
    print("  /context list - List context files")
    print("  /context clear - Clear context files")
    print("  /lsp start <language> [path] - Start LSP server")
    print("  /kb add <key> <content> - Add to knowledge base")
    print("  /kb search <query> - Search knowledge base")
    print("  /agents - Check subagent status")
    print("  /server <url> - Set Ollama server URL")
    print("  /notify setup <topic> - Enable ntfy.sh notifications")
    print("  /notify off - Disable notifications")
    print("  /notify test - Send test notification")
    print("\nWhen notifications are enabled, send messages to your topic to interact remotely!\n")
    
    # Add system context to initial prompt
    system_context = f"""Operating System: {sys.platform}
Current Directory: {os.getcwd()}
Home Directory: {os.path.expanduser('~')}
User: {os.environ.get('USER', 'unknown')}

{SYSTEM_PROMPT}"""
    
    # Add MCP tools to system prompt
    if MCP_TOOLS:
        mcp_tools_desc = "\n".join([f"- {name}: {tool['description']}" for name, tool in MCP_TOOLS.items()])
        system_context = system_context.replace("{mcp_tools_list}", mcp_tools_desc)
    else:
        system_context = system_context.replace("{mcp_tools_list}", "None connected")
    
    messages = [{"role": "system", "content": system_context}]
    
    # Auto-connect MCP servers from config
    config = load_config()
    for name, server_config in config.get("mcp_servers", {}).items():
        result = connect_mcp_server(name, server_config["command"], server_config.get("args"))
        print(f"[{result}]")
    
    # Start ntfy listener if enabled
    if config.get("ntfy", {}).get("enabled"):
        topic = config["ntfy"]["topic"]
        print(f"[Starting ntfy listener for topic: {topic}]")
        listener_thread = threading.Thread(target=ntfy_listener, args=(messages,), daemon=True)
        listener_thread.start()
    
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ['/quit', 'exit']:
            print("Goodbye!")
            break
            
        if user_input.lower() == '/clear':
            system_context = f"""Operating System: {sys.platform}
Current Directory: {os.getcwd()}
Home Directory: {os.path.expanduser('~')}
User: {os.environ.get('USER', 'unknown')}

{SYSTEM_PROMPT}"""
            messages = [{"role": "system", "content": system_context}]
            print("History cleared.")
            continue
        
        if user_input.lower() == '/save':
            try:
                with open('chat_history.json', 'w') as f:
                    json.dump(messages, f, indent=2)
                print("History saved to chat_history.json")
            except Exception as e:
                print(f"Error saving history: {e}")
            continue
        
        if user_input.lower().startswith('/exclude '):
            model_to_exclude = user_input[9:].strip()
            config = load_config()
            if model_to_exclude not in config["excluded_models"]:
                config["excluded_models"].append(model_to_exclude)
                save_config(config)
                print(f"Excluded {model_to_exclude}. Restart to apply changes.")
            else:
                print(f"{model_to_exclude} is already excluded.")
            continue
        
        if user_input.lower() == '/list-excluded':
            config = load_config()
            excluded = config.get("excluded_models", [])
            if excluded:
                print(f"Excluded models: {', '.join(excluded)}")
            else:
                print("No models excluded.")
            continue
        
        if user_input.lower().startswith('/save'):
            parts = user_input.split(maxsplit=1)
            session_name = parts[1] if len(parts) > 1 else None
            session_file = save_session(messages, session_name)
            print(f"Session saved to {session_file}")
            continue
        
        if user_input.lower().startswith('/load '):
            session_name = user_input[6:].strip()
            loaded = load_session(session_name)
            if loaded:
                messages = loaded
                print(f"Session '{session_name}' loaded.")
            else:
                print(f"Session '{session_name}' not found.")
            continue
        
        if user_input.lower() == '/sessions':
            sessions = list_sessions()
            if sessions:
                print("Available sessions:")
                for s in sessions[:10]:  # Show last 10
                    print(f"  {s}")
            else:
                print("No saved sessions.")
            continue
        
        if user_input.lower().startswith('/model '):
            new_model = user_input[7:].strip()
            if new_model in AVAILABLE_MODELS:
                MODEL = new_model
                print(f"Switched to model: {MODEL}")
            else:
                print(f"Model '{new_model}' not available.")
            continue
        
        if user_input.lower() == '/models':
            print("Available models:")
            for m in AVAILABLE_MODELS:
                print(f"  {m}")
            continue
        
        if user_input.lower() == '/retry':
            if len(messages) > 1:
                # Remove last assistant response and retry
                if messages[-1]["role"] == "assistant":
                    messages.pop()
                user_input = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
                if not user_input:
                    print("Nothing to retry.")
                    continue
                messages.pop()  # Remove user message, will be re-added
            else:
                print("Nothing to retry.")
                continue
        
        if user_input.lower().startswith('/run '):
            filepath = user_input[5:].strip()
            result = execute_bash(f"python3 {filepath}" if filepath.endswith('.py') else filepath)
            print(result)
            continue
        
        if user_input.lower().startswith('/config git '):
            parts = user_input[12:].split()
            if len(parts) >= 3:
                config = load_config()
                config["git"] = {
                    "remote_url": parts[0],
                    "username": parts[1],
                    "password": parts[2]
                }
                save_config(config)
                print("Git configuration saved.")
            else:
                print("Usage: /config git <url> <username> <password>")
            continue
        
        if user_input.lower().startswith('/config gitea '):
            parts = user_input[14:].split()
            if len(parts) >= 2:
                config = load_config()
                config["gitea"] = {
                    "url": parts[0],
                    "token": parts[1]
                }
                save_config(config)
                print("Gitea configuration saved.")
            else:
                print("Usage: /config gitea <url> <token>")
            continue
        
        if user_input.lower().startswith('/export '):
            filename = user_input[8:].strip()
            try:
                with open(filename, 'w') as f:
                    f.write("# Ollama CLI Conversation\n\n")
                    for msg in messages[1:]:  # Skip system prompt
                        role = msg["role"].upper()
                        content = msg["content"]
                        f.write(f"## {role}\n\n{content}\n\n")
                print(f"Conversation exported to {filename}")
            except Exception as e:
                print(f"Error exporting: {e}")
            continue
        
        if user_input.lower().startswith('/mcp connect '):
            parts = user_input[13:].split()
            if len(parts) >= 2:
                name = parts[0]
                command = parts[1]
                args = parts[2:] if len(parts) > 2 else None
                result = connect_mcp_server(name, command, args)
                print(result)
                # Save to config
                config = load_config()
                config["mcp_servers"][name] = {"command": command, "args": args}
                save_config(config)
            else:
                print("Usage: /mcp connect <name> <command> [args...]")
            continue
        
        if user_input.lower() == '/mcp list':
            if MCP_SERVERS:
                print("Connected MCP servers:")
                for server in MCP_SERVERS:
                    tools = [t for t in MCP_TOOLS if t.startswith(f"mcp_{server}_")]
                    print(f"  {server}: {len(tools)} tools")
                    for tool_key in tools[:5]:  # Show first 5
                        tool = MCP_TOOLS[tool_key]
                        print(f"    - {tool['name']}: {tool['description'][:60]}")
            else:
                print("No MCP servers connected.")
            continue
        
        if user_input.lower().startswith('/mcp disconnect '):
            name = user_input[16:].strip()
            if name in MCP_SERVERS:
                MCP_SERVERS[name].terminate()
                del MCP_SERVERS[name]
                # Remove tools
                MCP_TOOLS.clear()
                for key in list(MCP_TOOLS.keys()):
                    if key.startswith(f"mcp_{name}_"):
                        del MCP_TOOLS[key]
                print(f"Disconnected from MCP server '{name}'")
            else:
                print(f"MCP server '{name}' not connected.")
            continue
        
        if user_input.lower().startswith('/context '):
            arg = user_input[9:].strip()
            
            if arg == 'list':
                if SESSION_CONTEXT["context_files"]:
                    print("Context files:")
                    for cf in SESSION_CONTEXT["context_files"]:
                        print(f"  {cf['path']} ({len(cf['content'])} chars)")
                else:
                    print("No context files loaded.")
                continue
            
            if arg == 'clear':
                SESSION_CONTEXT["context_files"].clear()
                print("Context files cleared.")
                continue
            
            # Load file into context
            filepath = os.path.expanduser(arg)
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                SESSION_CONTEXT["context_files"].append({
                    "path": filepath,
                    "content": content
                })
                print(f"Added {filepath} to context ({len(content)} chars)")
            except Exception as e:
                print(f"Error loading context file: {e}")
            continue
        
        if user_input.lower().startswith('/lsp start '):
            parts = user_input[11:].split()
            if len(parts) >= 1:
                language = parts[0]
                path = parts[1] if len(parts) > 1 else "."
                result = start_lsp_server(language, path)
                print(result)
            else:
                print("Usage: /lsp start <language> [path]")
            continue
        
        if user_input.lower().startswith('/kb add '):
            parts = user_input[8:].split(maxsplit=1)
            if len(parts) == 2:
                key, content = parts
                result = kb_add(key, content)
                print(result)
            else:
                print("Usage: /kb add <key> <content>")
            continue
        
        if user_input.lower().startswith('/kb search '):
            query = user_input[11:].strip()
            result = kb_search(query)
            print(result)
            continue
        
        if user_input.lower() == '/agents':
            result = check_subagents()
            print(result)
            continue
        
        if user_input.lower().startswith('/server '):
            url = user_input[8:].strip()
            if not url.startswith('http'):
                url = f"http://{url}"
            if '/api/chat' not in url:
                url = f"{url}/api/chat"
            
            config = load_config()
            config["ollama_url"] = url
            save_config(config)
            print(f"Ollama server set to: {url}")
            print("Restart to apply changes.")
            continue
        
        if user_input.lower().startswith('/notify '):
            parts = user_input[8:].split(maxsplit=1)
            if len(parts) >= 1:
                cmd = parts[0]
                if cmd == 'setup' and len(parts) == 2:
                    topic = parts[1].strip()
                    config = load_config()
                    config["ntfy"] = {
                        "enabled": True,
                        "topic": topic,
                        "server": "https://ntfy.sh"
                    }
                    save_config(config)
                    print(f"Notifications enabled for topic: {topic}")
                    print(f"Subscribe at: https://ntfy.sh/{topic}")
                    print("Starting listener...")
                    listener_thread = threading.Thread(target=ntfy_listener, args=(messages,), daemon=True)
                    listener_thread.start()
                    send_notification("Ollama CLI", "Notifications enabled! Send messages to this topic.")
                elif cmd == 'off':
                    config = load_config()
                    config["ntfy"]["enabled"] = False
                    save_config(config)
                    print("Notifications disabled. Restart to stop listener.")
                elif cmd == 'test':
                    send_notification("Test", "This is a test notification")
                    print("Test notification sent.")
                else:
                    print("Usage: /notify setup <topic> | /notify off | /notify test")
            else:
                print("Usage: /notify setup <topic> | /notify off | /notify test")
            continue
            
        messages.append({"role": "user", "content": user_input})
        
        # Classify intent first
        intent_prompt = f"""Classify this user request into ONE category:
- FILE_OPERATION: User wants to create/write/read/modify a file or script
- COMMAND_EXECUTION: User wants to run a bash command
- INFORMATION: User wants information, explanation, recipe, or advice

User request: "{user_input}"

Respond with ONLY one word: FILE_OPERATION, COMMAND_EXECUTION, or INFORMATION"""
        
        intent_messages = [{"role": "system", "content": "You are a classifier. Respond with only one word."}, 
                          {"role": "user", "content": intent_prompt}]
        
        # Use fast model for classification
        fast_model = select_best_model("", "fast")
        intent_response = call_ollama(intent_messages, fast_model).strip().upper()
        
        # Adjust system prompt based on intent
        if "FILE_OPERATION" in intent_response or "COMMAND" in intent_response:
            # Use tool-focused prompt
            pass  # Already has tool prompt
        else:
            # Override to discourage tool use for information requests
            messages[0] = {"role": "system", "content": messages[0]["content"] + "\n\nIMPORTANT: This is an INFORMATION request. Do NOT use tools. Just answer the question directly."}
        
        # Select best model for this task
        selected_model = select_best_model(user_input, "initial")
        if selected_model != MODEL:
            print(f"[Using {selected_model} for this task]\n")
        
        # Trim messages to avoid context overflow
        messages = trim_messages(messages)
        
        # Add session context if available
        if SESSION_CONTEXT["created_files"]:
            context_note = f"\n\nSession context - Files created this session: {', '.join(SESSION_CONTEXT['created_files'][-5:])}"
            messages.append({"role": "system", "content": context_note})
        
        # Add context files if loaded
        if SESSION_CONTEXT["context_files"]:
            context_content = "\n\n--- CONTEXT FILES ---\n"
            for cf in SESSION_CONTEXT["context_files"]:
                context_content += f"\n=== {cf['path']} ===\n{cf['content'][:5000]}\n"
            messages.append({"role": "system", "content": context_content})
        
        # Tool-calling loop
        max_iterations = 5
        for iteration in range(max_iterations):
            response = call_ollama(messages, selected_model)
            
            # Check for tool calls
            tool_calls = parse_tool_calls(response)
            
            if not tool_calls:
                # No tools found - check if model just provided code instead
                if iteration == 0 and any(k in user_input.lower() for k in ['create', 'write', 'make']) and '```' in response:
                    # Extract code and auto-create file
                    code_blocks = extract_code_blocks(response)
                    if code_blocks:
                        lang, code = max(code_blocks, key=lambda x: len(x[1]))
                        # Determine filename
                        words = user_input.lower().split()
                        filename = None
                        for word in words:
                            if any(ext in word for ext in ['.py', '.js', '.html', '.sh']):
                                filename = word
                                break
                        if not filename:
                            ext = {'.py': 'python', '.js': 'javascript', '.sh': 'bash'}.get(lang, '.txt')
                            for key, val in {'.py': 'python', '.js': 'javascript', '.sh': 'bash'}.items():
                                if val == lang:
                                    ext = key
                                    break
                            if 'calculator' in user_input.lower():
                                filename = f'calculator{ext}'
                            else:
                                filename = f'script{ext}'
                        
                        print(f"\n[Model didn't use tool - auto-creating {filename}]")
                        result = write_file(filename, code)
                        print(f"[{result}]")
                        messages.append({"role": "assistant", "content": response})
                        messages.append({"role": "user", "content": f"Tool: write_file\nResult:\n{result}"})
                        continue
                
                # No tools needed, show response
                if '<tool_call>' in response or any(tool in response for tool in TOOLS.keys()):
                    print(f"\n[Warning: Model attempted tool use but format was incorrect]")
                
                messages.append({"role": "assistant", "content": response})
                
                # Remove the old auto-save logic since we handle it above now
                break
            
            # Execute tools
            print()  # Newline before tool execution
            tool_results = []
            for tool_name, params in tool_calls:
                # Check if it's an MCP tool
                if tool_name.startswith("mcp_") and tool_name in MCP_TOOLS:
                    mcp_tool = MCP_TOOLS[tool_name]
                    server_name = mcp_tool["server"]
                    actual_tool_name = mcp_tool["name"]
                    print(f"[Executing MCP tool: {actual_tool_name} on {server_name}]")
                    result = call_mcp_tool(server_name, actual_tool_name, params)
                    print(f"[Result: {result[:200]}{'...' if len(result) > 200 else ''}]")
                    tool_results.append(f"Tool: {tool_name}\nResult:\n{result}")
                elif tool_name in TOOLS:
                    print(f"[Executing: {tool_name}({json.dumps(params)[:100]}...)]")
                    result = TOOLS[tool_name](**params)
                    # Show more for git list operations
                    display_limit = 1000 if tool_name == "git_operation" else 200
                    print(f"[Result: {result[:display_limit]}{'...' if len(result) > display_limit else ''}]")
                    tool_results.append(f"Tool: {tool_name}\nResult:\n{result}")
                    
                    # Send notifications for key events
                    if tool_name == "write_file" and "Successfully" in result:
                        send_notification("File Created", f"Created: {params.get('path', 'file')}")
                    elif tool_name == "execute_bash" and "Exit code: 0" in result:
                        send_notification("Command Complete", f"Executed: {params.get('command', 'command')[:50]}")
                else:
                    tool_results.append(f"Error: Unknown tool '{tool_name}'")
            
            # Add assistant message and tool results
            messages.append({"role": "assistant", "content": response})
            
            # Check if there was a syntax error
            has_syntax_error = any('syntax error' in r.lower() for r in tool_results)
            
            # For successful file creation, confirm and stop
            if any('write_file' in r and 'Successfully' in r for r in tool_results):
                print("\nFile created successfully.")
                
                # Check if it's an executable script and offer to run it
                created_files = [params.get('path') for name, params in tool_calls if name == 'write_file']
                for filepath in created_files:
                    if filepath and (filepath.endswith('.py') or filepath.endswith('.sh')):
                        try:
                            response = input(f"\nRun {filepath}? (y/n): ").strip().lower()
                            if response == 'y':
                                print(f"\n[Running {filepath}...]")
                                result = execute_bash(f"python3 {filepath}" if filepath.endswith('.py') else f"bash {filepath}")
                                print(result)
                        except:
                            pass
                
                messages.append({"role": "user", "content": "\n\n".join(tool_results)})
                break
            
            # For other successful tool executions, show result and stop (don't loop)
            if not has_syntax_error and iteration == 0:
                print()  # Newline after tool results
                messages.append({"role": "user", "content": "\n\n".join(tool_results)})
                break
            
            # If syntax error, let model see it and try to fix
            if has_syntax_error:
                print("\n[Syntax error detected - asking model to fix]")
                messages.append({"role": "user", "content": "\n\n".join(tool_results) + "\n\nPlease fix the syntax error and try again."})
            else:
                messages.append({"role": "user", "content": "\n\n".join(tool_results)})
            
            # Switch to a general model for interpreting tool results
            response_model = select_best_model("", "tool_response")
            if response_model != selected_model:
                print(f"[Switching to {response_model} for response]\n")
                selected_model = response_model
            
            # If last iteration, get final response
            if iteration == max_iterations - 1:
                final_response = call_ollama(messages, selected_model)
                messages.append({"role": "assistant", "content": final_response})
        
        # Auto-save session after each interaction
        config = load_config()
        if config.get("auto_save_sessions", True):
            save_session(messages, "autosave")

if __name__ == "__main__":
    main()
