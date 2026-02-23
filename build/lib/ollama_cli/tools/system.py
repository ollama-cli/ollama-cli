import subprocess
import os
from .base import tool

@tool(
    name="execute_bash",
    description="Execute a bash command and return its output",
    parameters={
        "command": {"type": "string", "description": "The command to execute"}
    }
)
def execute_bash(command: str) -> str:
    # Basic safety check
    dangerous = ['rm -rf /', 'dd if=', 'mkfs', ':(){:|:&};:']
    if any(d in command for d in dangerous):
        return f"⚠️ DANGEROUS COMMAND BLOCKED: {command}"
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=os.getcwd(), 
            timeout=60
        )
        output = result.stdout + result.stderr
        return f"Exit code: {result.returncode}\n{output}" if output else f"Exit code: {result.returncode} (no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds"
    except Exception as e:
        return f"Error: {str(e)}"

@tool(
    name="get_env_info",
    description="Get information about the current system environment",
    parameters={}
)
def get_env_info() -> str:
    import sys
    import platform
    return f"""OS: {platform.system()} {platform.release()}
Python: {sys.version}
Shell: {os.environ.get('SHELL', 'unknown')}
User: {os.environ.get('USER', 'unknown')}
CWD: {os.getcwd()}"""
