"""
Startup script for Railway deployment.
Downloads the ONNX model from GitHub Releases before starting the app.
"""
import os
import urllib.request
import sys

MODEL_DIR = "./output/finbert-lora"
MODEL_URLS = {
    "sentiment_model.onnx": "https://github.com/AtharvVatsal/lz-quant/releases/download/v1.0.0/sentiment_model.onnx",
    "tokenizer.json": "https://github.com/AtharvVatsal/lz-quant/releases/download/v1.0.0/tokenizer.json",
    "tokenizer_config.json": "https://github.com/AtharvVatsal/lz-quant/releases/download/v1.0.0/tokenizer_config.json",
}

def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_path = os.path.join(MODEL_DIR, "sentiment_model.onnx")
    
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return
    
    print("Downloading ONNX model from GitHub Releases...")
    for filename, url in MODEL_URLS.items():
        filepath = os.path.join(MODEL_DIR, filename)
        if os.path.exists(filepath):
            print(f"  {filename} already exists")
            continue
        print(f"  Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"  ✓ {filename} downloaded")
        except Exception as e:
            print(f"  ✗ Failed to download {filename}: {e}")
            if filename == "sentiment_model.onnx":
                print("WARNING: Model download failed. Running in simulated mode.")
                return
    
    print("Model download complete!")

if __name__ == "__main__":
    download_model()
