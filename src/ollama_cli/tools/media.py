import subprocess
import os
import sys
import requests
import json
import random
import uuid
import time
from .base import tool
from ..core.config import load_config, update_config
from ..ui.formatter import print_status, print_error, console


def _is_comfy_running(comfy_url: str) -> bool:
    """Check if ComfyUI is reachable."""
    try:
        requests.get(comfy_url + "/queue", timeout=3)
        return True
    except Exception:
        return False


def _find_comfy_main(comfy_path: str):
    """Return the path to ComfyUI's main.py if it exists."""
    expanded = os.path.expanduser(comfy_path)
    main_py = os.path.join(expanded, "main.py")
    if os.path.exists(main_py):
        return main_py
    return None


def _prompt_comfy_path():
    """Ask the user for the ComfyUI install path."""
    from prompt_toolkit import prompt as pt_prompt
    try:
        path = pt_prompt("Enter ComfyUI path (or empty to cancel): ").strip()
    except (KeyboardInterrupt, EOFError):
        return None
    if not path:
        return None
    expanded = os.path.expanduser(path)
    if not os.path.exists(os.path.join(expanded, "main.py")):
        print_error(f"No main.py found in {expanded}. Not a valid ComfyUI directory.")
        return None
    # Save so we never ask again
    update_config("comfy_path", path)
    print_status(f"ComfyUI path saved to config: {path}")
    return expanded


def _start_comfy(config: dict) -> bool:
    """Auto-start ComfyUI in the background. Prompts for path if needed."""
    comfy_path = config.get("comfy_path", "~/ComfyUI")
    main_py = _find_comfy_main(comfy_path)

    if not main_py:
        print_error(f"ComfyUI not found at {os.path.expanduser(comfy_path)}")
        resolved = _prompt_comfy_path()
        if not resolved:
            return False
        comfy_path = resolved
        main_py = os.path.join(resolved, "main.py")

    comfy_url = config.get("comfy_url", "http://127.0.0.1:8188").rstrip("/")
    expanded = os.path.expanduser(comfy_path)
    print_status(f"Starting ComfyUI from [bold]{expanded}[/bold]...")

    # Parse port from comfy_url
    try:
        from urllib.parse import urlparse
        port = urlparse(comfy_url).port or 8188
    except Exception:
        port = 8188

    subprocess.Popen(
        [sys.executable, main_py, "--listen", "127.0.0.1", "--port", str(port)],
        cwd=expanded,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for it to become ready
    for i in range(30):
        time.sleep(2)
        if _is_comfy_running(comfy_url):
            print_status("ComfyUI is ready.")
            return True
        if i % 5 == 4:
            print_status("Still waiting for ComfyUI to start...")

    print_error("ComfyUI failed to start within 60 seconds.")
    return False


def _ensure_comfy(config: dict) -> bool:
    """Ensure ComfyUI is running, auto-starting if needed."""
    comfy_url = config.get("comfy_url", "http://127.0.0.1:8188").rstrip("/")
    if _is_comfy_running(comfy_url):
        return True
    return _start_comfy(config)

@tool(
    name="get_comfy_status",
    description="Check the status of image generation jobs or the ComfyUI queue",
    parameters={
        "prompt_id": {"type": "string", "description": "Optional: Specific prompt ID to check", "default": ""}
    }
)
def get_comfy_status(prompt_id: str = "") -> str:
    """Queries the ComfyUI API for queue or history status"""
    config = load_config()
    if not _ensure_comfy(config):
        return "Error: ComfyUI is not running and could not be started."
    comfy_url = config.get("comfy_url", "http://127.0.0.1:8188").rstrip("/")
    
    try:
        if prompt_id:
            # Check history for specific prompt
            resp = requests.get(f"{comfy_url}/history/{prompt_id}", timeout=5)
            if resp.status_code == 200:
                history = resp.json()
                if prompt_id in history:
                    return f"Status for {prompt_id}: Completed. Result saved."
            
            # Check queue if not in history
            resp = requests.get(f"{comfy_url}/queue", timeout=5)
            if resp.status_code == 200:
                queue_data = resp.json()
                for job in queue_data.get("queue_running", []):
                    if job[1] == prompt_id: return f"Status for {prompt_id}: Currently Running."
                for job in queue_data.get("queue_pending", []):
                    if job[1] == prompt_id: return f"Status for {prompt_id}: Pending in Queue."
            
            return f"Status for {prompt_id}: ID not found in current queue or history."
        else:
            # General queue status
            resp = requests.get(f"{comfy_url}/queue", timeout=5)
            if resp.status_code == 200:
                queue_data = resp.json()
                running = len(queue_data.get("queue_running", []))
                pending = len(queue_data.get("queue_pending", []))
                return f"ComfyUI Queue Status: {running} running, {pending} pending."
            return f"Error: Could not retrieve queue from {comfy_url}"
    except Exception as e:
        return f"Error connecting to ComfyUI: {str(e)}"

@tool(
    name="generate_image",
    description="Generate an image using the local Stable Diffusion workflow (ComfyUI)",
    parameters={
        "prompt": {"type": "string", "description": "The image description"},
        "width": {"type": "integer", "description": "Image width in pixels (default: 512)", "optional": True},
        "height": {"type": "integer", "description": "Image height in pixels (default: 512)", "optional": True}
    }
)
def generate_image(prompt: str, width: int = 512, height: int = 512) -> str:
    """Invokes the ComfyUI API with WebSocket for realtime progress tracking"""
    config = load_config()
    if not _ensure_comfy(config):
        return "Error: ComfyUI is not running and could not be started."
    comfy_url = config.get("comfy_url", "http://127.0.0.1:8188").rstrip("/")
    image_model = config.get("image_model", "dreamshaper_8_pruned.safetensors")
    
    # Override defaults for SDXL models if user didn't specify
    if "xl" in image_model.lower() and width == 512 and height == 512:
        width, height = (1024, 1024)

    client_id = str(uuid.uuid4())
    
    try:
        import websocket
    except ImportError:
        return "Error: 'websocket-client' not installed. Please run 'pip install websocket-client'."

    try:
        # Define the workflow directly
        workflow = {
            "1": {
                "inputs": {"ckpt_name": image_model},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": f"masterpiece, best quality, highly detailed, {prompt}",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "text": "blurry, low quality, distorted, watermark, nsfw",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {"width": width, "height": height, "batch_size": 1},
                "class_type": "EmptyLatentImage"
            },
            "5": {
                "inputs": {
                    "seed": random.randint(1, 1000000000), "steps": 20, "cfg": 7.0,
                    "sampler_name": "dpmpp_2m", "scheduler": "karras",
                    "denoise": 1.0, "model": ["1", 0],
                    "positive": ["2", 0], "negative": ["3", 0],
                    "latent_image": ["4", 0]
                },
                "class_type": "KSampler"
            },
            "6": {
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
                "class_type": "VAEDecode"
            },
            "7": {
                "inputs": {
                    "filename_prefix": "ollama_cli_",
                    "images": ["6", 0]
                },
                "class_type": "SaveImage"
            }
        }
        
        # Connect to WebSocket
        ws_url = comfy_url.replace("http://", "ws://").replace("https://", "wss://") + f"/ws?clientId={client_id}"
        ws = websocket.WebSocket()
        ws.connect(ws_url)
        
        # Queue the prompt
        payload = {"prompt": workflow, "client_id": client_id}
        response = requests.post(f"{comfy_url}/prompt", json=payload, timeout=10)
        
        if response.status_code != 200:
            return f"Error from ComfyUI: {response.status_code} - {response.text}"
            
        data = response.json()
        prompt_id = data.get('prompt_id')
        
        print_status(f"Realtime monitoring active (Prompt ID: {prompt_id})")
        
        found_file = None
        from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Queued...", total=20) # 20 steps default
            
            while True:
                out = ws.recv()
                if not out: break
                
                if isinstance(out, str):
                    msg = json.loads(out)
                    if msg['type'] == 'progress':
                        p_data = msg['data']
                        if p_data['prompt_id'] == prompt_id:
                            progress.update(task, description="[green]Sampling...", completed=p_data['value'], total=p_data['max'])
                    
                    if msg['type'] == 'executing':
                        e_data = msg['data']
                        if e_data['prompt_id'] == prompt_id:
                            node = e_data['node']
                            if node is None: # Execution finished
                                break
                            # Update description based on node type if possible (or just generic)
                            progress.update(task, description=f"[yellow]Node {node}...")
                    
                    if msg['type'] == 'executed':
                        ex_data = msg['data']
                        if ex_data['prompt_id'] == prompt_id:
                            # We can extract the filename here!
                            if 'output' in ex_data and 'images' in ex_data['output']:
                                images = ex_data['output']['images']
                                if images:
                                    img_name = images[0]['filename']
                                    output_path = os.path.expanduser(config.get("comfy_output_path", "~/ComfyUI/output"))
                                    found_file = os.path.join(output_path, img_name)
                else:
                    continue # Binary data (previews) ignored for now
        
        ws.close()
        
        if found_file and os.path.exists(found_file):
            if sys.platform == "darwin":
                subprocess.run(["open", found_file])
            else:
                subprocess.run(["xdg-open", found_file])
            return f"Image generation complete!\nSaved to: {found_file}\nOpening image now..."
        
        return f"Image generation finished, but could not locate file automatically. Prompt ID: {prompt_id}"
            
    except Exception as e:
        return f"Error generating image: {str(e)}"

@tool(
    name="analyze_image",
    description="Analyze or describe a local image file using AI vision",
    parameters={
        "image_path": {"type": "string", "description": "Path to the image file"},
        "prompt": {"type": "string", "description": "What to ask about the image", "default": "Describe this image in detail."}
    }
)
def analyze_image(image_path: str, prompt: str = "Describe this image in detail.") -> str:
    """Invokes the Ollama vision API"""
    config = load_config()
    vision_model = config.get("vision_model", "llama3.2-vision")
    
    from ..core.ollama import OllamaClient
    client = OllamaClient(config.get("ollama_url"))
    
    print_status(f"Analyzing image with {vision_model}...")
    return client.describe_image(image_path, prompt, model=vision_model)

@tool(
    name="speak_text",
    description="Convert text to speech using Piper TTS (high quality)",
    parameters={
        "text": {"type": "string", "description": "Text to speak"},
        "output_file": {"type": "string", "description": "Output filename", "default": "output.wav"}
    }
)
def speak_text(text: str, output_file: str = "output.wav") -> str:
    """Uses the best available TTS engine (Kokoro > Piper > System)"""
    config = load_config()
    
    # 1. Try Kokoro (Ultra-Realistic)
    kokoro_python = os.path.expanduser(config.get("kokoro_path", "~/.ollama-cli/kokoro/venv/bin/python"))
    if os.path.exists(kokoro_python):
        try:
            # Inline script to run in Kokoro venv
            kokoro_script = f"""
import sys
import soundfile as sf
import os
try:
    from kokoro import KPipeline
    import torch
except ImportError:
    print("Dependencies missing")
    sys.exit(1)

text = sys.argv[1]
output_file = sys.argv[2]

# Initialize pipeline
pipeline = KPipeline(lang_code='a') 
generator = pipeline(text, voice='af_bella', speed=1)

all_audio = []
for gs, ps, audio in generator:
    if audio is not None:
        all_audio.append(audio)

if all_audio:
    import numpy as np
    full_audio = np.concatenate(all_audio)
    sf.write(output_file, full_audio, 24000)
    print("Success")
else:
    print("No audio generated")
    sys.exit(1)
"""
            # Run in venv
            subprocess.run([kokoro_python, "-c", kokoro_script, text, output_file], 
                           check=True, capture_output=True)
            
            if sys.platform == "darwin":
                subprocess.run(["afplay", output_file])
            return f"Audio saved to {output_file} and played (using Ultra-Realistic Kokoro TTS)"
        except Exception as e:
            # Continue to fallback
            pass

    # 2. Try Piper (High-Quality)
    piper_config = config.get("piper", {})
    piper_bin = os.path.expanduser(piper_config.get("path", "~/.ollama-cli/piper/piper"))
    model_path = os.path.expanduser(piper_config.get("model", "~/.ollama-cli/piper/voice.onnx"))
    
    if os.path.exists(piper_bin) and os.path.exists(model_path):
        try:
            # Piper reads from stdin and writes to file
            cmd = f'echo "{text}" | {piper_bin} --model {model_path} --output_file {output_file}'
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
            
            # Auto-play on macOS
            if sys.platform == "darwin":
                subprocess.run(["afplay", output_file])
                
            return f"Audio saved to {output_file} and played (using high-quality Piper TTS)"
        except Exception as e:
            pass
    
    # 3. Fallback to system 'say' command on macOS
    if sys.platform == "darwin":
        try:
            subprocess.run(["say", "-o", output_file, text], check=True)
            subprocess.run(["afplay", output_file])
            return f"Audio saved to {output_file} and played (using system 'say' fallback)"
        except Exception as e:
            return f"Error using 'say': {e}"
            
    return "TTS engine not found. Please run the installer to set up Piper or Kokoro."
