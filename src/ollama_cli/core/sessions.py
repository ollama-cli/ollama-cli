"""
Session management for ollama-cli.

Each session is a directory under ~/.ollama-cli/sessions/<session_id>/ containing:
  - state.json   — messages, model, checkpoints, metadata
  - mailbox/     — subagent mailbox files (copied on save)
"""

import os
import json
import shutil
import uuid
from datetime import datetime
from typing import List, Dict, Optional

SESSION_BASE = os.path.expanduser("~/.ollama-cli/sessions")
# Legacy path for backwards compat with list_sessions
_LEGACY_DIR = os.path.expanduser("~/.ollama-cli-sessions")


def _safe_name(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in (' ', '.', '_', '-')).strip()


def generate_session_id() -> str:
    """Generate a short unique session ID."""
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{now}_{short}"


def _session_dir(session_id: str) -> str:
    return os.path.join(SESSION_BASE, _safe_name(session_id))


def save_session(messages: List[Dict], session_id: Optional[str] = None,
                 model: str = "", auto_model: bool = True,
                 checkpoints: list = None) -> str:
    """Save full session state to a session directory. Returns the session ID."""
    if not session_id:
        session_id = generate_session_id()

    sdir = _session_dir(session_id)
    os.makedirs(sdir, exist_ok=True)

    # Save state
    state = {
        "session_id": session_id,
        "messages": messages,
        "model": model,
        "auto_model": auto_model,
        "checkpoints": checkpoints or [],
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(sdir, "state.json"), "w") as f:
        json.dump(state, f, indent=2)

    # Mailbox dir lives inside the session dir — nothing to copy.
    # The Mailbox instance writes directly to <session_dir>/mailbox/.

    return session_id


def load_session(session_id: str) -> Optional[Dict]:
    """Load full session state. Returns dict with messages, model, checkpoints, etc.

    Returns None if session not found.
    """
    sdir = _session_dir(session_id)
    state_file = os.path.join(sdir, "state.json")

    if not os.path.exists(state_file):
        # Try legacy single-file format
        return _load_legacy(session_id)

    with open(state_file, "r") as f:
        state = json.load(f)

    # Mailbox dir is inside the session dir — Mailbox instance will
    # read from it directly when initialized with this session's path.

    return state


def _load_legacy(session_id: str) -> Optional[Dict]:
    """Load from old single-file format for backwards compat."""
    for base in [_LEGACY_DIR, SESSION_BASE]:
        path = os.path.join(base, f"{session_id}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            return {
                "session_id": session_id,
                "messages": data.get("messages", []),
                "model": "",
                "auto_model": True,
                "checkpoints": [],
            }
    return None


def list_sessions() -> List[str]:
    """List available session IDs, newest first."""
    sessions = []
    if os.path.exists(SESSION_BASE):
        for name in os.listdir(SESSION_BASE):
            sdir = os.path.join(SESSION_BASE, name)
            if os.path.isdir(sdir) and os.path.exists(os.path.join(sdir, "state.json")):
                sessions.append(name)
    # Also include legacy sessions
    if os.path.exists(_LEGACY_DIR):
        for f in os.listdir(_LEGACY_DIR):
            if f.endswith(".json"):
                sessions.append(f[:-5])
    return sorted(sessions, reverse=True)


def delete_session(session_id: str) -> bool:
    """Delete a session."""
    sdir = _session_dir(session_id)
    if os.path.exists(sdir):
        shutil.rmtree(sdir)
        return True
    # Try legacy
    legacy = os.path.join(_LEGACY_DIR, f"{session_id}.json")
    if os.path.exists(legacy):
        os.remove(legacy)
        return True
    return False


def get_mailbox_dir(session_id: str) -> str:
    """Return the mailbox directory path for a session."""
    return os.path.join(_session_dir(session_id), "mailbox")


def get_session_preview(session_id: str) -> str:
    """Get a short preview of a session (first user message)."""
    state = load_session(session_id)
    if not state:
        return "(empty)"
    for msg in state.get("messages", []):
        if msg.get("role") == "user" and not msg["content"].startswith("Tool Result"):
            preview = msg["content"][:50]
            if len(msg["content"]) > 50:
                preview += "..."
            return preview
    return "(no messages)"
