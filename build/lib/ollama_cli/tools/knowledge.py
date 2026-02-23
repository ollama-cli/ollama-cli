import json
import os
from .base import tool

KB_FILE = os.path.expanduser("~/.ollama-cli-kb.json")

def load_kb():
    if os.path.exists(KB_FILE):
        with open(KB_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_kb(kb):
    with open(KB_FILE, 'w') as f:
        json.dump(kb, f, indent=2)

@tool(
    name="kb_add",
    description="Add content to the knowledge base",
    parameters={
        "key": {"type": "string", "description": "Search key"},
        "content": {"type": "string", "description": "Information content"}
    }
)
def kb_add(key: str, content: str) -> str:
    kb = load_kb()
    kb[key] = content
    save_kb(kb)
    return f"Added '{key}' to knowledge base"

@tool(
    name="kb_search",
    description="Search the knowledge base",
    parameters={
        "query": {"type": "string", "description": "Search query"}
    }
)
def kb_search(query: str) -> str:
    kb = load_kb()
    results = [f"{k}: {v[:100]}..." for k, v in kb.items() if query.lower() in k.lower() or query.lower() in v.lower()]
    return "\n".join(results) if results else "No matches found"
