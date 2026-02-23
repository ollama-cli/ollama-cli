import subprocess
import json
import os
from typing import Dict, Any, Optional, List

MCP_SERVERS = {}
MCP_TOOLS = {}

def connect_mcp_server(name: str, command: str, args: Optional[List[str]] = None) -> str:
    """Connect to an MCP server via stdio"""
    try:
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
                "clientInfo": {"name": "ollama-cli", "version": "3.0"}
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

def call_mcp_tool(server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
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
