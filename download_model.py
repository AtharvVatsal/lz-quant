"""
Startup script for Railway deployment.
Downloads the ONNX model from HuggingFace in background.
"""
import os
import subprocess
import sys

MODEL_DIR = "./output/finbert-lora"
HF_BASE = "https://huggingface.co/datasets/atharvvatsal-av/lz-quant-model/resolve/main"

MODEL_FILES = [
    "sentiment_model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "adapter_config.json",
]

def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_path = os.path.join(MODEL_DIR, "sentiment_model.onnx")
    
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return True
    
    print("Downloading ONNX model from HuggingFace...")
    
    for filename in MODEL_FILES:
        filepath = os.path.join(MODEL_DIR, filename)
        if os.path.exists(filepath):
            print(f"  {filename} already exists")
            continue
        url = f"{HF_BASE}/{filename}"
        print(f"  Downloading {filename}...")
        try:
            result = subprocess.run(
                ["curl", "-L", "-o", filepath, url],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                print(f"  OK {filename}")
            else:
                print(f"  FAILED {filename}")
        except Exception as e:
            print(f"  FAILED {filename}: {e}")
            if filename == "sentiment_model.onnx":
                return False
    
    return True

if __name__ == "__main__":
    success = download_model()
    if not success:
        print("WARNING: Model download failed. Running in simulated mode.")
    print("Starting dual.py...")
    os.system("python dual.py")
