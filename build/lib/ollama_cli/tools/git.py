import subprocess
import requests
from typing import Optional
from .base import tool
from ..core.config import load_config

@tool(
    name="git_operation",
    description="Perform git operations (status, diff, commit, push, pull, list)",
    parameters={
        "operation": {"type": "string", "description": "Operation to perform (status|diff|commit|push|pull|list)"},
        "message": {"type": "string", "description": "Commit message", "default": "Auto commit"}
    }
)
def git_operation(operation: str, message: str = "Auto commit") -> str:
    config = load_config()
    git_config = config.get("git", {})
    gitea_config = config.get("gitea", {})
    
    try:
        if operation == "status":
            result = subprocess.run("git status", shell=True, capture_output=True, text=True)
        elif operation == "diff":
            result = subprocess.run("git diff", shell=True, capture_output=True, text=True)
        elif operation == "commit":
            result = subprocess.run(f'git add -A && git commit -m "{message}"', 
                                  shell=True, capture_output=True, text=True)
        elif operation == "push":
            result = subprocess.run("git push", shell=True, capture_output=True, text=True)
        elif operation == "pull":
            result = subprocess.run("git pull", shell=True, capture_output=True, text=True)
        elif operation == "list":
            if gitea_config.get("url") and gitea_config.get("token"):
                gitea_url = gitea_config["url"].rstrip("/")
                token = gitea_config["token"]
                response = requests.get(
                    f"{gitea_url}/api/v1/user/repos",
                    headers={"Authorization": f"token {token}"},
                    timeout=10
                )
                if response.status_code == 200:
                    repos = response.json()
                    return "\n".join([f"- {r['full_name']} ({r['clone_url']})" for r in repos])
                return f"Gitea API error: {response.status_code}"
            result = subprocess.run("find . -maxdepth 2 -name '.git' -type d | sed 's|/.git||'", 
                                  shell=True, capture_output=True, text=True)
        else:
            return f"Unknown git operation: {operation}"
        
        return result.stdout + result.stderr
    except Exception as e:
        return f"Git error: {str(e)}"
