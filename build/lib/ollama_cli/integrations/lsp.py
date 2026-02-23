import subprocess
import os
import json
from typing import Dict, Any, Optional

LSP_SERVERS = {}

def start_lsp_server(language: str, root_path: str = ".") -> str:
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

def lsp_goto_definition(language: str, file_path: str, line: int, character: int) -> str:
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
        
        return "Definition lookup sent (response parsing not implemented)"
    except Exception as e:
        return f"LSP error: {str(e)}"
