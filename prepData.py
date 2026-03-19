"""
LZ-QUANT — Dataset Preparation Script
Downloads and prepares real financial sentiment data for training.

The Financial PhraseBank on Hugging Face uses a deprecated loading script.
This script bypasses that by pulling from working sources and combining
multiple free datasets into one clean training file.

Sources (all free, no API keys):
  1. Financial PhraseBank — via direct Parquet/CSV download
  2. Twitter Financial News Sentiment — zeroshot/twitter-financial-news-sentiment
  3. (Optional) CryptoBERT dataset — crypto-specific sentiment

Output:
  ./data/financial_sentiment.csv
  Columns: sentence, label (0=Bearish, 1=Neutral, 2=Bullish)

Usage:
  python prepare_data.py
  python prepare_data.py --include-crypto    # Also include crypto dataset

Then train:
  python train.py  (it will auto-detect ./data/financial_sentiment.csv)
"""

import os
import sys
import csv
import argparse
import json
from pathlib import Path

try:
    from datasets import load_dataset, Dataset
    import numpy as np
except ImportError:
    sys.exit("Missing: pip install datasets numpy")


DATA_DIR = "./data"
OUTPUT_FILE = os.path.join(DATA_DIR, "financial_sentiment.csv")

LABEL_MAP_DISPLAY = {0: "Bearish", 1: "Neutral", 2: "Bullish"}


def download_financial_phrasebank():
    """
    Download Financial PhraseBank using the Parquet files directly.
    This bypasses the broken loading script.
    """
    print("\n[1/3] Financial PhraseBank")

    samples = []

    # Strategy 1: Use the working mirror (atrost/financial_phrasebank)
    # This is a clean Parquet re-upload that doesn't have the broken loading script.
    try:
        print("     Source: atrost/financial_phrasebank (working Parquet mirror)")
        ds = load_dataset("atrost/financial_phrasebank")

        for split_name in ds:
            for row in ds[split_name]:
                samples.append({
                    "sentence": row["sentence"],
                    "label": row["label"],  # 0=negative, 1=neutral, 2=positive
                })

        print(f"     ✓ Loaded {len(samples)} samples from mirror")
        return samples

    except Exception as e:
        print(f"     Mirror load failed: {e}")

    # Strategy 2: Try the original with trust_remote_code=False
    try:
        print("     Trying original source (takala/financial_phrasebank)...")
        ds = load_dataset(
            "takala/financial_phrasebank",
            "sentences_allagree",
            trust_remote_code=False,
        )
        for row in ds["train"]:
            samples.append({
                "sentence": row["sentence"],
                "label": row["label"],
            })
        print(f"     ✓ Loaded {len(samples)} samples")
        return samples

    except Exception as e2:
        print(f"     Original source failed: {e2}")

    # Strategy 3: Try the enhanced version
    try:
        print("     Trying enhanced version (descartes100/enhanced-financial-phrasebank)...")
        ds = load_dataset("descartes100/enhanced-financial-phrasebank")

        for split_name in ds:
            for row in ds[split_name]:
                text = row.get("sentence", row.get("text", ""))
                label = row.get("label", row.get("sentiment", 1))

                # Normalize labels to 0=Bearish, 1=Neutral, 2=Bullish
                if isinstance(label, str):
                    label_map = {"negative": 0, "neutral": 1, "positive": 2}
                    label = label_map.get(label.lower(), 1)

                if text and len(text) > 5:
                    samples.append({"sentence": text, "label": int(label)})

        print(f"     ✓ Loaded {len(samples)} samples from enhanced version")
        return samples

    except Exception as e3:
        print(f"     Enhanced version failed: {e3}")

    print("!!Could not load Financial PhraseBank from any source")
    return samples


def download_twitter_financial():
    """
    Download Twitter Financial News Sentiment dataset.
    This one always works — it's a standard Parquet dataset.

    Original labels: 0=Bearish, 1=Bullish, 2=Neutral
    We remap to:     0=Bearish, 1=Neutral, 2=Bullish (matching our model)
    """
    print("\n[2/3] Twitter Financial News Sentiment")
    print("     Source: zeroshot/twitter-financial-news-sentiment")

    samples = []

    try:
        ds = load_dataset("zeroshot/twitter-financial-news-sentiment")

        # This dataset has train and validation splits
        for split_name in ["train", "validation"]:
            if split_name in ds:
                for row in ds[split_name]:
                    text = row["text"].strip()
                    original_label = row["label"]

                    # Remap: original 0=Bearish, 1=Bullish, 2=Neutral
                    # Our format: 0=Bearish, 1=Neutral, 2=Bullish
                    label_remap = {0: 0, 1: 2, 2: 1}
                    label = label_remap.get(original_label, 1)

                    if len(text) > 10:  # Skip very short tweets
                        samples.append({
                            "sentence": text,
                            "label": label,
                        })

        print(f"     @ Loaded {len(samples)} samples")

    except Exception as e:
        print(f"     !!Failed: {e}")

    return samples


def download_crypto_sentiment():
    """
    Download CryptoBERT crypto-specific sentiment dataset.
    Great for training if you're primarily trading crypto.

    Labels: 0=Bearish, 1=Neutral, 2=Bullish (already matches our format)
    """
    print("\n[3/3] CryptoBERT Crypto Sentiment")
    print("     Source: ElKulako/cryptobert")

    samples = []

    try:
        ds = load_dataset("ElKulako/cryptobert", split="train")

        for row in ds:
            text = row.get("text", "").strip()
            label = row.get("label", row.get("sentiment", 1))

            # CryptoBERT labels: 0=Bearish, 1=Neutral, 2=Bullish
            if isinstance(label, str):
                label_map = {"Bearish": 0, "Neutral": 1, "Bullish": 2,
                             "negative": 0, "neutral": 1, "positive": 2}
                label = label_map.get(label, 1)

            if len(text) > 10:
                samples.append({
                    "sentence": text[:512],  # Trim very long posts
                    "label": int(label),
                })

        print(f"     @Loaded {len(samples)} samples")

    except Exception as e:
        print(f"     !!Failed: {e}")
        print(f"     (This dataset is optional — core training works without it)")

    return samples


def validate_and_clean(samples):
    """Remove duplicates, empty strings, and validate labels."""
    seen = set()
    clean = []

    for s in samples:
        text = s["sentence"].strip()
        label = s["label"]

        # Validate
        if not text or len(text) < 5:
            continue
        if label not in (0, 1, 2):
            continue
        if text in seen:
            continue

        seen.add(text)
        clean.append({"sentence": text, "label": label})

    return clean


def save_dataset(samples, output_path):
    """Save as CSV that train.py can load directly."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "label"])
        writer.writeheader()
        writer.writerows(samples)

    print(f"\n[SAVED] {output_path}")
    return output_path


def print_stats(samples):
    """Print dataset statistics."""
    labels = [s["label"] for s in samples]
    total = len(samples)

    counts = {0: labels.count(0), 1: labels.count(1), 2: labels.count(2)}
    lengths = [len(s["sentence"]) for s in samples]

    print(f"\n{'─' * 60}")
    print(f"  DATASET STATISTICS")
    print(f"{'─' * 60}")
    print(f"  Total samples  : {total:,}")
    print(f"  Bearish (0)    : {counts[0]:>6,}  ({counts[0]/total*100:.1f}%)")
    print(f"  Neutral (1)    : {counts[1]:>6,}  ({counts[1]/total*100:.1f}%)")
    print(f"  Bullish (2)    : {counts[2]:>6,}  ({counts[2]/total*100:.1f}%)")
    print(f"  Avg length     : {np.mean(lengths):.0f} chars")
    print(f"  Min length     : {min(lengths)} chars")
    print(f"  Max length     : {max(lengths)} chars")
    print(f"{'─' * 60}")

    # Print a few samples from each class
    print(f"\n  Sample sentences:")
    for label_id, label_name in LABEL_MAP_DISPLAY.items():
        class_samples = [s for s in samples if s["label"] == label_id]
        if class_samples:
            example = class_samples[0]["sentence"][:80]
            print(f"  [{label_name:>7}] \"{example}...\"")

def main():
    parser = argparse.ArgumentParser(description="LZ-Quant Dataset Preparation")
    parser.add_argument("--include-crypto", action="store_true",
                        help="Include CryptoBERT crypto sentiment data")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE,
                        help="Output CSV path")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  LZ-QUANT — Dataset Preparation")
    print(f"{'='*60}")
    print(f"  Downloading free financial sentiment datasets...")

    all_samples = []
    # Source 1: Financial PhraseBank
    fpb_samples = download_financial_phrasebank()
    all_samples.extend(fpb_samples)
    # Source 2: Twitter Financial News
    twitter_samples = download_twitter_financial()
    all_samples.extend(twitter_samples)
    # Source 3: CryptoBERT (optional)
    if args.include_crypto:
        crypto_samples = download_crypto_sentiment()
        all_samples.extend(crypto_samples)
    else:
        print("\n[3/3] CryptoBERT Crypto Sentiment")
        print("     Skipped (use --include-crypto to include)")
    if not all_samples:
        print("\n[ERROR] No data downloaded. Check your internet connection.")
        sys.exit(1)
    # Clean and deduplicate
    print(f"\n[CLEAN] Removing duplicates and validating...")
    clean_samples = validate_and_clean(all_samples)
    print(f"[CLEAN] {len(all_samples):,} raw → {len(clean_samples):,} clean samples")
    # Shuffle
    np.random.seed(42)
    indices = np.random.permutation(len(clean_samples))
    clean_samples = [clean_samples[i] for i in indices]
    # Stats
    print_stats(clean_samples)
    # Save
    save_dataset(clean_samples, args.output)

    print(f"\nDataset ready! Now train with:")
    print(f"  python train.py")

if __name__ == "__main__":
    main()