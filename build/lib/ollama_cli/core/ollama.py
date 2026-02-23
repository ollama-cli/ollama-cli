import json
import requests
import subprocess
import sys
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

    def get_available_models(self) -> List[str]:
        """Get list of models available in Ollama"""
        try:
            # Try API first
            response = requests.get(self.list_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m['name'] for m in data.get('models', [])]
            
            # Fallback to CLI
            result = subprocess.run("ollama list", shell=True, capture_output=True, text=True, timeout=5)
            models = []
            for line in result.stdout.split('\n')[1:]:  # Skip header
                if line.strip():
                    models.append(line.split()[0])
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
            }, stream=stream, timeout=60)
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
        """Select best model based on task type"""
        if not available_models:
            return default_model
            
        # Model preferences
        code_models = ["qwen2.5-coder", "mistral-nemo", "codellama", "deepseek-coder"]
        general_models = ["llama3.2", "llama3.1", "mistral"]
        
        # Simple heuristic
        code_keywords = ['code', 'script', 'program', 'function', 'debug', 'python', 'javascript']
        is_code_task = any(k in user_input.lower() for k in code_keywords)
        
        target_list = code_models if is_code_task else general_models
        
        for target in target_list:
            for available in available_models:
                if target in available.lower():
                    return available
                    
        return available_models[0] if available_models else default_model
