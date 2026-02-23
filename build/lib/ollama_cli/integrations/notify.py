import requests
import json
import threading
from typing import Callable, List, Dict

def send_notification(config: Dict, title: str, message: str, priority: str = "default"):
    """Send notification via ntfy.sh"""
    ntfy_config = config.get("ntfy", {})
    
    if not ntfy_config.get("enabled") or not ntfy_config.get("topic"):
        return
    
    try:
        server = ntfy_config.get("server", "https://ntfy.sh")
        topic = ntfy_config["topic"]
        url = f"{server}/{topic}"
        
        requests.post(url, 
            data=message.encode('utf-8'),
            headers={
                "Title": title,
                "Priority": priority
            },
            timeout=5
        )
    except:
        pass

def ntfy_listener(config: Dict, on_message: Callable[[str], str]):
    """Listen for ntfy notifications and process them"""
    ntfy_config = config.get("ntfy", {})
    
    if not ntfy_config.get("enabled") or not ntfy_config.get("topic"):
        return
    
    server = ntfy_config.get("server", "https://ntfy.sh")
    topic = ntfy_config["topic"]
    url = f"{server}/{topic}/json"
    
    try:
        response = requests.get(url, stream=True, timeout=None)
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if data.get("event") == "message":
                    title = data.get("title", "")
                    message = data.get("message", "")
                    
                    # Skip messages sent by this script
                    if title in ["Response", "Ollama CLI", "File Created", "Command Complete"]:
                        continue
                    
                    # Process message
                    response_text = on_message(message)
                    
                    # Send response back
                    send_notification(config, "Response", response_text[:1000])
    except:
        pass

def start_notify_thread(config: Dict, on_message: Callable[[str], str]):
    thread = threading.Thread(target=ntfy_listener, args=(config, on_message), daemon=True)
    thread.start()
    return thread
