import subprocess
import os
import sys
import requests
import json
import random
from .base import tool
from ..core.config import load_config

@tool(
    name="generate_image",
    description="Generate an image using the local Stable Diffusion workflow (ComfyUI)",
    parameters={
        "prompt": {"type": "string", "description": "The image description"}
    }
)
def generate_image(prompt: str) -> str:
    """Invokes the ComfyUI API directly with a predefined workflow"""
    config = load_config()
    comfy_url = config.get("comfy_url", "http://127.0.0.1:8188").rstrip("/")
    image_model = config.get("image_model", "dreamshaper_8_pruned.safetensors")
    
    # Determine dimensions (SDXL prefers 1024x1024)
    width, height = (512, 512)
    if "xl" in image_model.lower():
        width, height = (1024, 1024)

    try:
        # Check if ComfyUI is running
        try:
            requests.get(comfy_url, timeout=2)
        except:
            return f"Error: ComfyUI server ({comfy_url}) appears to be down. Please start it first or configure the URL with '/config comfy <url>'."

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
        
        payload = {"prompt": workflow}
        response = requests.post(f"{comfy_url}/prompt", json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            prompt_id = data.get('prompt_id', 'unknown')
            return f"Image generation queued successfully on {comfy_url}.\nPrompt ID: {prompt_id}\nCheck your ComfyUI output folder for 'ollama_cli_' prefixed files."
        else:
            return f"Error form ComfyUI: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error generating image: {str(e)}"

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
            
    return "No compatible TTS engine found. Please run the installer."
