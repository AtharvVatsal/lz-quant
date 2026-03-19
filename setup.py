from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lz-quant",
    version="1.0.0",
    author="LZ-Quant Team",
    description="Dual Market Sentiment Trading Engine with AI-powered sentiment analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/lz-quant",
    packages=find_packages(exclude=["lz-dashboard", "lzQuant", "output", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "websockets>=12.0",
        "aiohttp>=3.9.0",
        "python-dotenv>=1.0.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "numpy>=1.26.0",
    ],
    extras_require={
        "inference": [
            "onnx>=1.16.0",
            "onnxruntime>=1.16.0",
            "transformers>=4.40.0",
        ],
        "training": [
            "torch>=2.0.0",
            "transformers>=4.40.0",
            "datasets>=2.19.0",
            "peft>=0.11.0",
            "accelerate>=0.30.0",
            "scikit-learn>=1.4.0",
            "onnx>=1.16.0",
            "onnxruntime>=1.16.0",
        ],
        "gpu": [
            "onnxruntime-gpu>=1.18.0",
        ],
        "visualization": [
            "matplotlib>=3.8.0",
            "seaborn>=0.12.0",
        ],
        "all": [
            "torch>=2.0.0",
            "transformers>=4.40.0",
            "datasets>=2.19.0",
            "peft>=0.11.0",
            "accelerate>=0.30.0",
            "scikit-learn>=1.4.0",
            "onnx>=1.16.0",
            "onnxruntime>=1.16.0",
            "matplotlib>=3.8.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lz-quant=dual:main",
        ],
    },
)
