"""
LZ-QUANT — DistilBERT Fine-Tuning with LoRA + FP16
Target Hardware : Intel i5-13th Gen H | 32GB DDR5 | NVIDIA RTX 3050 (4GB VRAM)
Framework       : PyTorch + Hugging Face Transformers + PEFT (LoRA)
Dataset         : Financial PhraseBank (Malo et al., 2014)
Output          : Fine-tuned model  → ./output/finbert-lora/
                  ONNX export       → ./output/finbert-lora/sentiment_model.onnx

VRAM Budget Breakdown (approximate):
  DistilBERT base (FP16)        ~  250 MB
  LoRA adapter weights          ~   10 MB
  Optimizer states (AdamW)      ~   20 MB
  Activations (batch=8)         ~  200 MB
  PyTorch CUDA overhead         ~  500 MB
  Estimated peak                ~ 1.0 GB  (well within 4GB)

Install dependencies (all open-source, all free):
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install transformers datasets peft accelerate scikit-learn onnx onnxruntime
"""

import os
import json
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

# 1. CONFIGURATION — All hyperparameters in one place for easy experimentation
CONFIG = {
    # Model
    "model_name": "distilbert-base-uncased",
    "num_labels": 3,                    # Bullish (positive), Bearish (negative), Neutral
    "label_map": {                      # Human-readable label mapping
        0: "Bearish",
        1: "Neutral",
        2: "Bullish",
    },

    # LoRA — Low-Rank Adaptation
    # Instead of updating ALL 66M parameters of DistilBERT, LoRA injects
    # small trainable matrices into the attention layers. This reduces
    # trainable params to ~0.5% of the original, slashing VRAM usage.
    #
    # r (rank)          : Lower = fewer params, less VRAM. 8 is the sweet spot.
    # lora_alpha        : Scaling factor. Rule of thumb: alpha = 2 * r.
    # lora_dropout      : Regularization to prevent overfitting on small datasets.
    # target_modules    : Which layers get LoRA adapters. For DistilBERT,
    # "q_lin" and "v_lin" are the query and value projections inside multi-head attention.
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "lora_target_modules": ["q_lin", "v_lin"],

    # Training
    # Effective batch size = batch_size × gradient_accumulation_steps
    # Here: 8 × 4 = 32 effective batch size, but only 8 samples in VRAM at once.
    "batch_size": 8,
    "gradient_accumulation_steps": 4,   # Simulate batch_size=32 without the VRAM cost
    "learning_rate": 2e-4,              # Higher than normal fine-tuning because LoRA
    "weight_decay": 0.01,
    "num_epochs": 10,
    "warmup_ratio": 0.1,               # 10% of steps for LR warmup
    "max_seq_length": 128,              # Financial headlines are short; 128 tokens is plenty
    "fp16": True,                       # Mixed precision — halves memory for activations

    # Paths
    "output_dir": "./output/finbert-lora",
    "onnx_export_path": "./output/finbert-lora/sentiment_model.onnx",
}

# 2. DATASET LOADING — Financial PhraseBank
"""
DATASET FORMAT EXPLAINED:
─────────────────────────
Financial PhraseBank contains ~4,846 sentences from financial news, each
labeled by 5–8 annotators. We use the "sentences_allagree" subset (where
all annotators agreed on the sentiment) for cleaner labels.

Each sample looks like:
{
    "sentence": "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn ...",
    "label":    1  (0=negative/bearish, 1=neutral, 2=positive/bullish)
}

HOW TO USE YOUR OWN DATASET INSTEAD:
If you have your own CSV/JSON of financial text + sentiment labels, format
it as follows:

  Option A — CSV file (my_data.csv):
  ┌──────────────────────────────────────────────────────┬───────┐
  │ sentence                                             │ label │
  ├──────────────────────────────────────────────────────┼───────┤
  │ "Tesla beats Q3 earnings expectations"               │   2   │  ← Bullish
  │"Fed signals further rate hikes amid inflation fears" │   0   │  ← Bearish
  │ "Markets closed flat ahead of jobs report"           │   1   │  ← Neutral
  └──────────────────────────────────────────────────────┴───────┘

  Load with:
    from datasets import load_dataset
    dataset = load_dataset("csv", data_files="my_data.csv")["train"]

  Option B — Python list of dicts:
    data = [
        {"sentence": "BTC breaks $100k resistance",   "label": 2},
        {"sentence": "SEC delays ETF ruling again",   "label": 0},
        {"sentence": "Volume steady across exchanges", "label": 1},
    ]
    dataset = Dataset.from_list(data)

  Option C — JSON Lines file (my_data.jsonl):
    {"sentence": "NVIDIA announces record revenue", "label": 2}
    {"sentence": "Oil prices tumble on demand fears", "label": 0}

    dataset = load_dataset("json", data_files="my_data.jsonl")["train"]

Labels MUST be integers: 0 = Bearish, 1 = Neutral, 2 = Bullish
"""


def load_financial_phrasebank():
    """
    Load financial sentiment data. Priority order:
      1. Local prepared CSV (from prepare_data.py) — best option
      2. Hugging Face Financial PhraseBank — may fail on newer HF versions
      3. Built-in 30-sample demo — only for testing the pipeline
    """
    # Priority 1: Local prepared dataset
    local_csv = "./data/financial_sentiment.csv"
    if os.path.exists(local_csv):
        try:
            print(f"[DATA] Found local dataset: {local_csv}")
            dataset = load_dataset("csv", data_files=local_csv)["train"]
            print(f"[DATA] Loaded {len(dataset)} samples from prepared dataset")
            return dataset
        except Exception as e:
            print(f"[DATA] Local CSV load failed: {e}")

    print("[DATA] No local dataset found at ./data/financial_sentiment.csv")
    print("[DATA] Run 'python prepare_data.py' first to download real training data!")
    print("[DATA] Attempting Hugging Face download as fallback...\n")

    # Priority 2: Hugging Face download
    try:
        print("[DATA] Downloading Financial PhraseBank (sentences_allagree)...")
        dataset = load_dataset(
            "takala/financial_phrasebank",
            "sentences_allagree",
        )
        raw = dataset["train"]
        print(f"[DATA] Loaded {len(raw)} samples from Financial PhraseBank")
        return raw
    except Exception as e1:
        print(f"[DATA] Standard load failed: {e1}")

    # Strategy 2: Try loading the Parquet version directly
    try:
        print("[DATA] Trying alternative dataset source...")
        dataset = load_dataset(
            "financial_phrasebank",
            "sentences_allagree",
        )
        raw = dataset["train"]
        print(f"[DATA] Loaded {len(raw)} samples from Financial PhraseBank (alt)")
        return raw
    except Exception as e2:
        print(f"[DATA] Alternative load failed: {e2}")

    # Strategy 3: Download raw CSV from HF and parse manually
    try:
        print("[DATA] Trying direct Parquet download...")
        dataset = load_dataset(
            "takala/financial_phrasebank",
            split="train",
        )
        print(f"[DATA] Loaded {len(dataset)} samples from Financial PhraseBank (direct)")
        return dataset
    except Exception as e3:
        print(f"[DATA] Direct download failed: {e3}")

    print("[DATA] All download methods failed. Using built-in demo dataset.")
    print("[DATA] This is only for testing — train on real data for production!")

    # Fallback demo dataset
    # 30 samples to verify the pipeline runs end-to-end.
    # Replace this with your own data for actual training.
    demo_data = [
        # Bullish (label=2)
        {"sentence": "Revenue surged 45% year-over-year, beating all analyst estimates", "label": 2},
        {"sentence": "The company announced a massive $10B share buyback program", "label": 2},
        {"sentence": "Strong demand in cloud services drove record quarterly earnings", "label": 2},
        {"sentence": "Bitcoin broke through key resistance levels on institutional buying", "label": 2},
        {"sentence": "Gross margins expanded by 300 basis points on pricing power", "label": 2},
        {"sentence": "The FDA granted breakthrough therapy designation to the drug candidate", "label": 2},
        {"sentence": "Order backlog grew to $50 billion, signaling sustained demand", "label": 2},
        {"sentence": "Free cash flow increased 60% enabling accelerated debt paydown", "label": 2},
        {"sentence": "Management raised full-year guidance above consensus expectations", "label": 2},
        {"sentence": "Strategic acquisition expected to be immediately accretive to earnings", "label": 2},
        # Bearish (label=0)
        {"sentence": "The company issued a profit warning citing weakening consumer demand", "label": 0},
        {"sentence": "Credit default swaps widened sharply on bankruptcy concerns", "label": 0},
        {"sentence": "Supply chain disruptions forced the factory to halt production", "label": 0},
        {"sentence": "Revenue missed estimates by 15% as core markets contracted", "label": 0},
        {"sentence": "The SEC launched a formal investigation into accounting practices", "label": 0},
        {"sentence": "Debt-to-equity ratio climbed to dangerous levels above 3.0x", "label": 0},
        {"sentence": "Key executive departures raised governance red flags", "label": 0},
        {"sentence": "Operating losses deepened as customer churn accelerated", "label": 0},
        {"sentence": "The stock was downgraded by three major brokerages simultaneously", "label": 0},
        {"sentence": "Covenant violations triggered technical default on senior notes", "label": 0},
        # Neutral (label=1)
        {"sentence": "The quarterly report was in line with market expectations", "label": 1},
        {"sentence": "Trading volume was average with no significant price movement", "label": 1},
        {"sentence": "The board appointed a new independent director effective next month", "label": 1},
        {"sentence": "The company maintained its existing dividend at $0.50 per share", "label": 1},
        {"sentence": "Analysts held their price targets steady following the update", "label": 1},
        {"sentence": "Markets await the Federal Reserve decision scheduled for Wednesday", "label": 1},
        {"sentence": "The merger remains under routine regulatory review with no issues", "label": 1},
        {"sentence": "Inventory levels normalized after seasonal adjustments", "label": 1},
        {"sentence": "The index closed marginally higher in thin holiday trading", "label": 1},
        {"sentence": "Currency hedging offset the impact of foreign exchange fluctuations", "label": 1},
    ]
    return Dataset.from_list(demo_data)

# 3. TOKENIZATION & DATALOADERS
def tokenize_and_split(dataset, tokenizer, config):
    """
    Tokenize all sentences and split into train/validation sets (80/20).
    Returns PyTorch DataLoaders ready for the training loop.
    """
    # Tokenize everything at once
    def tokenize_fn(batch):
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=config["max_seq_length"],
            return_tensors=None,  # Return lists; Dataset handles conversion
        )

    print("[TOKENIZER] Tokenizing dataset...")
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["sentence"])
    tokenized = tokenized.rename_column("label", "labels")

    # Train/val split (stratified to maintain class balance)
    # Cast labels to ClassLabel type if needed (fixes stratify compatibility)
    try:
        from datasets import ClassLabel
        label_feature = ClassLabel(num_classes=3, names=["Bearish", "Neutral", "Bullish"])
        tokenized = tokenized.cast_column("labels", label_feature)
        split = tokenized.train_test_split(test_size=0.2, seed=42, stratify_by_column="labels")
    except Exception:
        # Fallback: non-stratified split (still works, just slightly less balanced)
        print("[TOKENIZER] Note: Using non-stratified split (small dataset)")
        split = tokenized.train_test_split(test_size=0.2, seed=42)

    # Set torch format AFTER splitting (ensures labels are integer tensors)
    split["train"].set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    split["test"].set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # On Windows, num_workers > 0 often causes multiprocessing errors.
    # The dataset is small enough that parallel loading isn't needed anyway.
    import platform
    num_workers = 0 if platform.system() == "Windows" else 2

    train_loader = DataLoader(
        split["train"],
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,   # Speeds up CPU→GPU transfer
        num_workers=num_workers,
        drop_last=True,    # Avoid weird batch-size-1 edge cases
    )
    val_loader = DataLoader(
        split["test"],
        batch_size=config["batch_size"] * 2,  # Can use larger batch for eval (no gradients)
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    print(f"[TOKENIZER] Train: {len(split['train'])} samples | Val: {len(split['test'])} samples")
    return train_loader, val_loader

# 4. MODEL SETUP — DistilBERT + LoRA Injection
def build_model(config):
    """
    Load DistilBERT for 3-class classification, then wrap it with LoRA adapters.
    This freezes the base model and only trains the tiny LoRA matrices.
    """
    print(f"[MODEL] Loading {config['model_name']}...")

    # Load base model with 3 output classes
    model = DistilBertForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"],
        # NOTE: Do NOT load the model in FP16 here. Mixed precision training
        # requires FP32 base weights. autocast() handles FP16 during the forward
        # pass, and GradScaler manages the gradient scaling. Loading in FP16
        # would cause "Attempting to unscale FP16 gradients" errors.
    )

    # LoRA Configuration
    # This is the magic that makes fine-tuning fit in 4GB VRAM.
    #
    # How LoRA works (simplified):
    #   Normal weight update:  W_new = W_old + ΔW        (ΔW is huge: 768×768)
    #   LoRA weight update:    W_new = W_old + A × B      (A is 768×8, B is 8×768)
    #
    #   Instead of learning a 589,824-parameter update matrix,
    #   LoRA learns two small matrices with only 12,288 parameters total.
    #   That's a 48× reduction per layer.
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,             # Sequence classification task
        r=config["lora_r"],                     # Rank of decomposition (8)
        lora_alpha=config["lora_alpha"],         # Scaling factor (16)
        lora_dropout=config["lora_dropout"],     # Dropout on LoRA layers (0.1)
        target_modules=config["lora_target_modules"],  # ["q_lin", "v_lin"]
        bias="none",                            # Don't train bias terms
    )

    model = get_peft_model(model, lora_config)

    # Print parameter breakdown
    model.print_trainable_parameters()
    # Expected output: "trainable params: 296,451 || all params: 67,251,459 || trainable%: 0.4408"

    return model

# 5. TRAINING LOOP — With FP16 Mixed Precision & Gradient Accumulation
def train(model, train_loader, val_loader, config):
    """
    Custom training loop with:
      - FP16 mixed precision (autocast + GradScaler) to halve activation memory
      - Gradient accumulation to simulate large batches in limited VRAM
      - Linear LR warmup + decay schedule
      - Validation after each epoch with full classification report
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Using device: {device}")
    if device.type == "cuda":
        print(f"[TRAIN] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[TRAIN] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model.to(device)

    # Optimizer
    # Only optimize LoRA parameters (the rest are frozen)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Learning Rate Scheduler
    total_steps = len(train_loader) * config["num_epochs"] // config["gradient_accumulation_steps"]
    warmup_steps = int(total_steps * config["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Mixed Precision Setup
    # GradScaler prevents FP16 underflow by dynamically scaling the loss.
    # autocast() automatically runs matmuls in FP16 but keeps certain ops
    # (like loss computation and softmax) in FP32 for numerical stability.
    scaler = GradScaler("cuda", enabled=config["fp16"])

    # Training
    best_val_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    print(f"\n{'='*70}")
    print(f"  TRAINING CONFIG")
    print(f"  Epochs: {config['num_epochs']} | Batch: {config['batch_size']} | "
          f"Accum steps: {config['gradient_accumulation_steps']} | "
          f"Effective batch: {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"  LR: {config['learning_rate']} | Warmup: {warmup_steps}/{total_steps} steps")
    print(f"  FP16: {config['fp16']} | LoRA rank: {config['lora_r']}")
    print(f"{'='*70}\n")

    for epoch in range(config["num_epochs"]):
        #Train Phase
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass inside autocast context for FP16
            with autocast(device_type="cuda", enabled=config["fp16"]):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                # Scale loss by accumulation steps so the effective loss
                # magnitude stays the same regardless of how many steps
                # we accumulate over.
                loss = outputs.loss / config["gradient_accumulation_steps"]

            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * config["gradient_accumulation_steps"]

            # Only update weights every `gradient_accumulation_steps` steps
            if (step + 1) % config["gradient_accumulation_steps"] == 0:
                # Gradient clipping to prevent exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = epoch_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # Validation Phase
        val_loss, val_f1, report = evaluate(model, val_loader, device, config)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        # VRAM monitoring
        vram_used = ""
        if device.type == "cuda":
            allocated = torch.cuda.max_memory_allocated() / 1e9
            vram_used = f" | Peak VRAM: {allocated:.2f} GB"

        print(f"  Epoch {epoch+1}/{config['num_epochs']}  │  "
              f"Train Loss: {avg_train_loss:.4f}  │  "
              f"Val Loss: {val_loss:.4f}  │  "
              f"Val F1: {val_f1:.4f}{vram_used}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_checkpoint(model, config, history, epoch)
            print(f"  └─ New best model saved (F1: {val_f1:.4f})")

    print(f"\n{'='*70}")
    print(f"  Training complete. Best validation F1: {best_val_f1:.4f}")
    print(f"  Model saved to: {config['output_dir']}")
    print(f"{'='*70}")

    # Print final classification report
    print(f"\n  Final Validation Report:\n{report}")

    return model, history


def evaluate(model, val_loader, device, config):
    """Run validation and return loss, macro F1, and full classification report."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type="cuda", enabled=config["fp16"]):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    target_names = [config["label_map"][i] for i in range(config["num_labels"])]
    report = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    # Extract macro F1 for model selection
    f1 = float(classification_report(
        all_labels, all_preds,
        output_dict=True,
        zero_division=0,
    )["macro avg"]["f1-score"])

    return avg_loss, f1, report

# 6. CHECKPOINT & EXPORT
def save_checkpoint(model, config, history, epoch):
    """Save LoRA adapter weights and training metadata."""
    os.makedirs(config["output_dir"], exist_ok=True)

    # Save only the LoRA adapter (tiny — typically < 2 MB)
    model.save_pretrained(config["output_dir"])

    # Save training metadata
    meta = {
        "config": config,
        "best_epoch": epoch + 1,
        "history": history,
    }
    with open(os.path.join(config["output_dir"], "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


def export_to_onnx(model, tokenizer, config):
    """
    Export the fine-tuned model to ONNX format for fast CPU/GPU inference. ONNX models run ~2-3x faster than PyTorch for inference.
    """
    print("\n[EXPORT] Converting to ONNX format...")

    device = next(model.parameters()).device
    # Merge LoRA weights back into the base model for a clean single-file export
    merged_model = model.merge_and_unload()
    merged_model.eval()

    # Create dummy input matching real inference shape
    dummy_input = tokenizer(
        "The company reported strong quarterly earnings",
        padding="max_length",
        truncation=True,
        max_length=config["max_seq_length"],
        return_tensors="pt",
    ).to(device)

    os.makedirs(os.path.dirname(config["onnx_export_path"]), exist_ok=True)

    # Export with dynamic axes so batch size can vary at inference time
    torch.onnx.export(
        merged_model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        config["onnx_export_path"],
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # Verify the export
    import onnx
    onnx_model = onnx.load(config["onnx_export_path"])
    onnx.checker.check_model(onnx_model)

    file_size = os.path.getsize(config["onnx_export_path"]) / 1e6
    print(f"[EXPORT] ONNX model saved: {config['onnx_export_path']} ({file_size:.1f} MB)")
    print(f"[EXPORT] Use this with: onnxruntime.InferenceSession()")

    return config["onnx_export_path"]

# 7. INFERENCE DEMO — Test the trained model
def run_inference_demo(model, tokenizer, config):
    """Quick sanity check: run inference on sample financial headlines."""
    device = next(model.parameters()).device
    model.eval()

    test_sentences = [
        "Tesla stock surges 12% after record deliveries crush expectations",
        "Inflation fears mount as CPI data comes in hotter than expected",
        "Markets closed mixed in quiet trading session ahead of Fed minutes",
        "Bitcoin drops below key support level as whales dump holdings",
        "Company announces strategic partnership to expand into Asian markets",
    ]

    print("  INFERENCE DEMO")
    for sentence in test_sentences:
        inputs = tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=config["max_seq_length"],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad(), autocast(device_type="cuda", enabled=config["fp16"]):
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
        pred_label = config["label_map"][int(np.argmax(probs))]

        print(f"\n  \"{sentence[:70]}{'...' if len(sentence) > 70 else ''}\"")
        print(f"  → Bearish: {probs[0]:.4f}  |  Neutral: {probs[1]:.4f}  |  Bullish: {probs[2]:.4f}")
        print(f"  → Prediction: {pred_label} (confidence: {probs.max():.2%})")

# 8. MAIN — Orchestrate the full pipeline
def main():
    print("\n" + "="*70)
    print("  LZ-QUANT (Latency-Zero): Training")
    print("="*70 + "\n")

    # Device diagnostics
    if torch.cuda.is_available():
        print(f"[SYSTEM] CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"[SYSTEM] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Pre-emptive VRAM cleanup
        torch.cuda.empty_cache()
    else:
        print("[SYSTEM] WARNING: No CUDA GPU detected. Training will run on CPU (much slower).")

    # Step 1: Load dataset
    dataset = load_financial_phrasebank()

    # Step 2: Tokenize
    tokenizer = DistilBertTokenizerFast.from_pretrained(CONFIG["model_name"])
    train_loader, val_loader = tokenize_and_split(dataset, tokenizer, CONFIG)

    # Step 3: Build model
    model = build_model(CONFIG)

    # Step 4: Train
    model, history = train(model, train_loader, val_loader, CONFIG)

    # Step 5: Inference demo
    run_inference_demo(model, tokenizer, CONFIG)

    # Step 6: Export to ONNX
    try:
        export_to_onnx(model, tokenizer, CONFIG)
    except Exception as e:
        print(f"[EXPORT] ONNX export failed: {e}")
        print("[EXPORT] The PyTorch model is still saved and usable.")

    # Save tokenizer alongside model (needed for inference in)
    tokenizer.save_pretrained(CONFIG["output_dir"])
    print(f"\n[DONE] Tokenizer saved to {CONFIG['output_dir']}")
    print("[DONE] Ready for Real-Time Pipeline.")

if __name__ == "__main__":
    main()