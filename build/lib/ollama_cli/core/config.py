import os
import json

CONFIG_FILE = os.path.expanduser("~/.ollama-cli-config.json")

DEFAULT_CONFIG = {
    "excluded_models": [],
    "git": {
        "remote_url": "",
        "username": "",
        "password": ""
    },
    "gitea": {
        "url": "",
        "token": ""
    },
    "auto_save_sessions": True,
    "mcp_servers": {},
    "ollama_url": "http://localhost:11434",
    "comfy_url": "http://localhost:9000",
    "piper": {
        "path": "~/.ollama-cli/piper/piper",
        "model": "~/.ollama-cli/piper/voice.onnx"
    },
    "ntfy": {
        "enabled": False,
        "topic": "",
        "server": "https://ntfy.sh"
    },
    "default_model": "llama3.2"
}

def load_config():
    """Load config file"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Merge with defaults to ensure all keys exist
                for k, v in DEFAULT_CONFIG.items():
                    if k not in config:
                        config[k] = v
                return config
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()

def save_config(config):
    """Save config file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def update_config(key, value):
    """Update a specific config key"""
    config = load_config()
    config[key] = value
    save_config(config)
