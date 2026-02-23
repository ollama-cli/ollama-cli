#!/bin/bash

# Ollama CLI Installer
# Installs Ollama CLI, Ollama, and ComfyUI

set -e

echo "==========================================="
echo "      Ollama CLI All-in-One Installer      "
echo "==========================================="
echo ""

# 1. Check Prerequisites
echo "[*] Checking system..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

if ! command -v git &> /dev/null; then
    echo "Error: git is required but not found."
    exit 1
fi

# 2. Install Ollama CLI
echo ""
echo "[*] Installing Ollama CLI..."
# Ensure we are in the directory containing pyproject.toml
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

if [ -f "pyproject.toml" ]; then
    pip3 install .
    echo "✅ Ollama CLI installed."
else
    echo "Error: Could not find pyproject.toml. Make sure you run this script from the ollama-cli directory."
    exit 1
fi

# 3. Install Ollama (Optional)
echo ""
echo "-------------------------------------------"
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is already installed."
else
    read -p "Do you want to install Ollama (LLM Server)? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "[*] Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        echo "✅ Ollama installed."
    else
        echo "Skipping Ollama installation."
    fi
fi

# 4. Install ComfyUI (Optional)
echo ""
echo "-------------------------------------------"
read -p "Do you want to install ComfyUI (Image Generation)? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    INSTALL_DIR="$HOME/ComfyUI"
    if [ -d "$INSTALL_DIR" ]; then
        echo "⚠️  ComfyUI directory already exists at $INSTALL_DIR"
        read -p "Update existing installation? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cd "$INSTALL_DIR"
            git pull
            pip3 install -r requirements.txt
            echo "✅ ComfyUI updated."
        fi
    else
        echo "[*] Cloning ComfyUI to $INSTALL_DIR..."
        git clone https://github.com/comfyanonymous/ComfyUI.git "$INSTALL_DIR"
        cd "$INSTALL_DIR"
        echo "[*] Installing ComfyUI dependencies..."
        pip3 install -r requirements.txt
        echo "✅ ComfyUI installed."
    fi

    # 5. Download Image Generation Model
    echo ""
    echo "[*] Downloading Image Generation Model (DreamShaper 8)..."
    MODEL_DIR="$INSTALL_DIR/models/checkpoints"
    mkdir -p "$MODEL_DIR"
    MODEL_PATH="$MODEL_DIR/dreamshaper_8_pruned.safetensors"
    if [ ! -f "$MODEL_PATH" ]; then
        echo "    This may take a few minutes (approx. 2GB)..."
        curl -L "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors" -o "$MODEL_PATH"
        echo "✅ Model downloaded."
    else
        echo "✅ Model already exists."
    fi

    echo ""
    echo "[!] To use Image Generation, you must start ComfyUI:"
    echo "    cd $INSTALL_DIR"
    echo "    python3 main.py"
    
    # Update config to point to ComfyUI
    python3 -c "
import json
import os
config_path = os.path.expanduser('~/.ollama-cli-config.json')
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
else:
    config = {}
config['comfy_url'] = 'http://127.0.0.1:8188'
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
"
else
    echo "Skipping ComfyUI installation."
fi

# 6. Install Piper TTS
echo ""
echo "-------------------------------------------"
PIPER_DIR="$HOME/.ollama-cli/piper"
if [ -f "$PIPER_DIR/piper" ]; then
    echo "✅ Piper TTS is already installed in $PIPER_DIR"
elif command -v piper &> /dev/null; then
    echo "✅ Piper TTS found in your PATH."
else
    read -p "Do you want to install Piper TTS (High-Quality Voice)? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p "$PIPER_DIR"
        cd "$PIPER_DIR"
        
        echo "[*] Downloading Piper binary..."
        OS_TYPE=$(uname -s | tr '[:upper:]' '[:lower:]')
        ARCH_TYPE=$(uname -m)
        
        if [[ "$OS_TYPE" == "darwin" ]]; then
            if [[ "$ARCH_TYPE" == "arm64" ]]; then
                URL="https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_macos_aarch64.tar.gz"
            else
                URL="https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_macos_x86_64.tar.gz"
            fi
        else
            URL="https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz"
        fi
        
        curl -L "$URL" -o piper.tar.gz
        if [ ! -s piper.tar.gz ]; then
            echo "Error: Download failed."
        else
            tar -xzf piper.tar.gz
            mv piper/piper .
            mv piper/lib* . || true
            rm -rf piper piper.tar.gz
            echo "✅ Piper binary installed."
        fi

        if [ ! -f "voice.onnx" ]; then
            echo "[*] Downloading High-Quality Voice Model..."
            curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" -o voice.onnx
            curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" -o voice.onnx.json
            echo "✅ Voice model downloaded."
        fi
    fi
fi

# 7. Install Kokoro TTS (Optional - Ultra Realistic)
echo ""
echo "-------------------------------------------"
read -p "Do you want to install Kokoro TTS (Ultra-Realistic human voice)? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "[*] Installing Kokoro dependencies..."
    # Kokoro is best installed via pip in a venv
    KOKORO_DIR="$HOME/.ollama-cli/kokoro"
    mkdir -p "$KOKORO_DIR"
    python3 -m venv "$KOKORO_DIR/venv"
    source "$KOKORO_DIR/venv/bin/activate"
    pip install kokoro>=0.3.4 soundfile
    
    # Update config to prioritize Kokoro
    python3 -c "
import json
import os
config_path = os.path.expanduser('~/.ollama-cli-config.json')
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
else:
    config = {}
config['tts_engine'] = 'kokoro'
config['kokoro_path'] = os.path.expanduser('~/.ollama-cli/kokoro/venv/bin/python')
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
"
    echo "✅ Kokoro TTS installed and set as default."
    deactivate
else
    echo "Skipping Kokoro installation."
fi

echo ""
echo "==========================================="
echo "      Installation Complete! 🚀            "
echo "==========================================="
echo ""
echo "To start the CLI, run:"
echo "  ollama-cli"
echo ""
if command -v ollama &> /dev/null; then
    echo "Ensure Ollama is running:"
    echo "  ollama serve"
fi
echo ""
