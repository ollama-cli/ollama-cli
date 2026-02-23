import os
import json
from datetime import datetime
from typing import List, Dict, Optional

SESSION_DIR = os.path.expanduser("~/.ollama-cli-sessions")

def save_session(messages: List[Dict], session_name: Optional[str] = None) -> str:
    """Save conversation session"""
    os.makedirs(SESSION_DIR, exist_ok=True)
    if not session_name:
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure filename is safe
    safe_name = "".join([c for c in session_name if c.isalnum() or c in (' ', '.', '_', '-')]).strip()
    session_file = os.path.join(SESSION_DIR, f"{safe_name}.json")
    
    with open(session_file, 'w') as f:
        json.dump({
            "messages": messages, 
            "timestamp": datetime.now().isoformat(),
            "name": session_name
        }, f, indent=2)
    return session_file

def load_session(session_name: str) -> Optional[List[Dict]]:
    """Load conversation session"""
    session_file = os.path.join(SESSION_DIR, f"{session_name}.json")
    if not os.path.exists(session_file):
        # Try without extension if name doesn't have it
        if not session_name.endswith('.json'):
            session_file = os.path.join(SESSION_DIR, f"{session_name}.json")
            
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            data = json.load(f)
            return data.get("messages", [])
    return None

def list_sessions() -> List[str]:
    """List available sessions"""
    if not os.path.exists(SESSION_DIR):
        return []
    sessions = [f.replace('.json', '') for f in os.listdir(SESSION_DIR) if f.endswith('.json')]
    return sorted(sessions, reverse=True)

def delete_session(session_name: str) -> bool:
    """Delete a session"""
    session_file = os.path.join(SESSION_DIR, f"{session_name}.json")
    if os.path.exists(session_file):
        os.remove(session_file)
        return True
    return False
