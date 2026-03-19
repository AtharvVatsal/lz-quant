"""
Startup script for Railway deployment.
Downloads the ONNX model from HuggingFace before starting the app.
"""
import os
import urllib.request
import zipfile
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
        return
    
    print("Downloading ONNX model from HuggingFace...")
    for filename in MODEL_FILES:
        filepath = os.path.join(MODEL_DIR, filename)
        if os.path.exists(filepath):
            print(f"  {filename} already exists")
            continue
        url = f"{HF_BASE}/{filename}"
        print(f"  Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"  OK {filename}")
        except Exception as e:
            print(f"  FAILED {filename}: {e}")
            if filename == "sentiment_model.onnx":
                print("WARNING: Model download failed. Running in simulated mode.")
                return
    
    print("Model download complete!")

if __name__ == "__main__":
    download_model()
