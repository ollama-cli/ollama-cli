import subprocess
import os
import sys
from .base import tool

@tool(
    name="run_python",
    description="Execute a Python script or snippet",
    parameters={
        "code": {"type": "string", "description": "Python code to execute"},
        "args": {"type": "string", "description": "Optional arguments", "default": ""}
    }
)
def run_python(code: str, args: str = "") -> str:
    try:
        # Create a temp file
        temp_file = ".ollama-cli-exec.py"
        with open(temp_file, "w") as f:
            f.write(code)
        
        cmd = f"{sys.executable} {temp_file} {args}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        output = result.stdout + result.stderr
        
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        if result.returncode == 0:
            return f"Success:\n{output}" if output else "Success (no output)"
        else:
            return f"Error (Exit {result.returncode}):\n{output}"
            
    except Exception as e:
        return f"Execution Error: {str(e)}"