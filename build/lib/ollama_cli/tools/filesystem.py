import os
import subprocess
from .base import tool

@tool(
    name="read_file",
    description="Read the contents of a file",
    parameters={
        "path": {"type": "string", "description": "Path to the file"}
    }
)
def read_file(path: str) -> str:
    try:
        with open(os.path.expanduser(path), 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool(
    name="write_file",
    description="Write content to a file",
    parameters={
        "path": {"type": "string", "description": "Path to the file"},
        "content": {"type": "string", "description": "Content to write"}
    }
)
def write_file(path: str, content: str) -> str:
    try:
        expanded_path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(expanded_path), exist_ok=True)
        with open(expanded_path, 'w') as f:
            f.write(content)
        return f"Successfully written to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

@tool(
    name="list_directory",
    description="List contents of a directory",
    parameters={
        "path": {"type": "string", "description": "Path to the directory", "default": "."}
    }
)
def list_directory(path: str = ".") -> str:
    try:
        expanded = os.path.expanduser(path)
        files = os.listdir(expanded)
        return "\n".join(files) if files else "Directory is empty"
    except Exception as e:
        return f"Error listing directory: {str(e)}"

@tool(
    name="grep_search",
    description="Search for a regular expression pattern in file contents",
    parameters={
        "pattern": {"type": "string", "description": "The regex pattern to search for"},
        "path": {"type": "string", "description": "Directory to search in", "default": "."},
        "include": {"type": "string", "description": "Glob pattern for files to include", "default": "*"}
    }
)
def grep_search(pattern: str, path: str = ".", include: str = "*") -> str:
    try:
        cmd = f'grep -rE --include="{include}" "{pattern}" {path}'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20)
        return result.stdout if result.stdout else "No matches found."
    except Exception as e:
        return f"Error: {str(e)}"

@tool(
    name="get_tree",
    description="Show the directory structure as a tree",
    parameters={
        "path": {"type": "string", "description": "Root path", "default": "."},
        "depth": {"type": "integer", "description": "Max depth", "default": 2}
    }
)
def get_tree(path: str = ".", depth: int = 2) -> str:
    try:
        cmd = f"find {path} -maxdepth {depth} -not -path '*/.*'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error: {str(e)}"
