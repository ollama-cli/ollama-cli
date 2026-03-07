import json
import requests
import subprocess
import sys
import os
from typing import List, Dict, Optional, Generator

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        # Strip trailing slashes and common API paths if user provided them in config
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/api/chat"):
            self.base_url = self.base_url[:-9]
        elif self.base_url.endswith("/api"):
            self.base_url = self.base_url[:-4]
            
        self.chat_url = f"{self.base_url}/api/chat"
        self.list_url = f"{self.base_url}/api/tags"

    def get_available_models(self, include_vision: bool = False) -> List[str]:
        """Get list of models available in Ollama"""
        try:
            # Try API first
            response = requests.get(self.list_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [m['name'] for m in data.get('models', [])]
                if not include_vision:
                    models = [m for m in models if "vision" not in m.lower()]
                return models
            
            # Fallback to CLI
            result = subprocess.run("ollama list", shell=True, capture_output=True, text=True, timeout=5)
            models = []
            for line in result.stdout.split('\n')[1:]:  # Skip header
                if line.strip():
                    model_name = line.split()[0]
                    if include_vision or "vision" not in model_name.lower():
                        models.append(model_name)
            return models
        except Exception:
            return []

    def chat(self, messages: List[Dict], model: str, stream: bool = True) -> Generator[str, None, str]:
        """Call Ollama API with streaming"""
        try:
            response = requests.post(self.chat_url, json={
                "model": model,
                "messages": messages,
                "stream": stream
            }, stream=stream, timeout=600)
            response.raise_for_status()
            
            full_response = ""
            if stream:
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if "message" in chunk and "content" in chunk["message"]:
                            content = chunk["message"]["content"]
                            full_response += content
                            yield content
                return full_response
            else:
                data = response.json()
                return data.get("message", {}).get("content", "")
                
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}. Make sure it's running.")
        except Exception as e:
            raise Exception(f"Error calling Ollama: {str(e)}")

    def select_best_model(self, user_input: str, available_models: List[str], default_model: str) -> str:
        """Select best reasoning model based on task type"""
        if not available_models:
            return default_model
            
        # Model preferences for REASONING (prefer larger/smarter models)
        code_models = ["qwen2.5-coder", "mistral-nemo", "codellama", "deepseek-coder"]
        general_models = ["mistral", "llama3.1", "llama3.2:3b", "llama3.2:latest", "llama3", "llama3.2"]
        
        user_input_lower = user_input.lower()
        
        # Check if it's a coding task
        code_keywords = ['code', 'script', 'program', 'function', 'debug', 'python', 'javascript']
        is_code_task = any(k in user_input_lower for k in code_keywords)
        
        target_list = code_models if is_code_task else general_models
        
        # Filter out vision models
        reasoning_pool = [m for m in available_models if "vision" not in m.lower()]
        if not reasoning_pool:
            return available_models[0]

        for target in target_list:
            for available in reasoning_pool:
                if target in available.lower():
                    # If it's a 1b model, only pick it if no other preferred models are available
                    if ":1b" in available.lower() and len(reasoning_pool) > 1:
                        continue
                    return available
                    
        # Fallback to a non-vision model, trying to avoid 1b
        for available in reasoning_pool:
            if ":1b" not in available.lower():
                return available

        return reasoning_pool[0]

    def get_context_length(self, model: str) -> int:
        """Get the context window size for a model."""
        try:
            response = requests.post(f"{self.base_url}/api/show", json={"name": model}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                model_info = data.get("model_info", {})
                for key, value in model_info.items():
                    if "context_length" in key:
                        return int(value)
        except Exception:
            pass
        return 4096  # conservative default

    def describe_image(self, image_path: str, prompt: str = "What is in this image?", model: str = "llama3.2-vision") -> str:
        """Analyze an image using a vision model"""
        import base64
        
        # Auto-download model if missing
        available = self.get_available_models(include_vision=True)
        if not any(model in m for m in available):
            print(f"[*] Vision model '{model}' not found. Downloading...")
            try:
                subprocess.run(f"ollama pull {model}", shell=True, check=True)
            except Exception as e:
                return f"Error downloading vision model: {e}"

        try:
            expanded_path = os.path.expanduser(image_path)
            with open(expanded_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            
            response = requests.post(self.chat_url, json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [img_data]
                    }
                ],
                "stream": False
            }, timeout=600)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "Error: No description returned.")
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
