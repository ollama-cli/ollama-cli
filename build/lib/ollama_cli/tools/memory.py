import json
import os
from .base import tool

PROJECT_MEMORY_FILE = ".ollama-cli/memory.json"

def load_project_memory():
    if os.path.exists(PROJECT_MEMORY_FILE):
        try:
            with open(PROJECT_MEMORY_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_project_memory(memory):
    os.makedirs(os.path.dirname(PROJECT_MEMORY_FILE), exist_ok=True)
    with open(PROJECT_MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=2)

@tool(
    name="remember_fact",
    description="Store a key piece of information about the project",
    parameters={
        "fact": {"type": "string", "description": "The fact to remember (e.g., 'API key is in .env', 'Main entry point is src/index.ts')"}
    }
)
def remember_fact(fact: str) -> str:
    memory = load_project_memory()
    facts = memory.get("facts", [])
    if fact not in facts:
        facts.append(fact)
        memory["facts"] = facts
        save_project_memory(memory)
        return f"Remembered: {fact}"
    return "Fact already known."

@tool(
    name="recall_facts",
    description="Recall all stored facts about the project",
    parameters={}
)
def recall_facts() -> str:
    memory = load_project_memory()
    facts = memory.get("facts", [])
    if not facts:
        return "No project facts stored."
    return "Project Facts:\n" + "\n".join(f"- {f}" for f in facts)

@tool(
    name="clear_memory",
    description="Clear project memory",
    parameters={}
)
def clear_memory() -> str:
    if os.path.exists(PROJECT_MEMORY_FILE):
        os.remove(PROJECT_MEMORY_FILE)
        return "Project memory cleared."
    return "Memory was already empty."