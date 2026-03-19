import json
import os
import sys
import argparse
import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    sys.exit("Missing: pip install matplotlib")

try:
    import seaborn as sns
except ImportError:
    sys.exit("Missing: pip install seaborn")

ONNX_AVAILABLE = False
try:
    import onnxruntime as ort
    from transformers import DistilBertTokenizerFast
    ONNX_AVAILABLE = True
except ImportError:
    pass

OUTPUT_DIR = "./output/finbert-lora"
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
META_PATH = os.path.join(OUTPUT_DIR, "training_meta.json")
ONNX_PATH = os.path.join(OUTPUT_DIR, "sentiment_model.onnx")

LABEL_MAP = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
COLORS = {
    "bg": "#0a0f1a",
    "surface": "#111b2e",
    "grid": "#1a2540",
    "text": "#c8d6e5",
    "accent": "#448aff",
    "bullish": "#00e676",
    "bearish": "#ff1744",
    "neutral": "#ffab00",
    "loss_train": "#448aff",
    "loss_val": "#ff6e40",
    "f1": "#00e676",
}


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["surface"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.5,
        "grid.linewidth": 0.5,
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.facecolor": COLORS["surface"],
        "legend.edgecolor": COLORS["grid"],
        "legend.fontsize": 9,
        "font.family": "monospace",
        "font.size": 10,
        "savefig.facecolor": COLORS["bg"],
        "savefig.edgecolor": COLORS["bg"],
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def load_training_meta():
    if not os.path.exists(META_PATH):
        print(f"[ERROR] Training metadata not found at {META_PATH}")
        print(f"[ERROR] Run train.py first to generate training results.")
        sys.exit(1)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    print(f"[LOAD] Training metadata loaded from {META_PATH}")
    print(f"[LOAD] Best epoch: {meta.get('best_epoch', '?')}")
    return meta


def plot_loss_curves(ax, history):
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"],
            color=COLORS["loss_train"], linewidth=2, marker="o",
            markersize=6, label="Train Loss", zorder=3)
    ax.plot(epochs, history["val_loss"],
            color=COLORS["loss_val"], linewidth=2, marker="s",
            markersize=6, label="Val Loss", zorder=3)
    ax.fill_between(epochs, history["train_loss"], history["val_loss"],
                    alpha=0.1, color=COLORS["accent"])
    best_idx = history["val_loss"].index(min(history["val_loss"]))
    ax.axvline(x=best_idx + 1, color=COLORS["bullish"], linestyle="--",
               alpha=0.5, linewidth=1)
    ax.annotate(f"Best: {min(history['val_loss']):.4f}",
                xy=(best_idx + 1, min(history["val_loss"])),
                xytext=(10, 15), textcoords="offset points",
                fontsize=8, color=COLORS["bullish"],
                arrowprops=dict(arrowstyle="->", color=COLORS["bullish"], lw=0.8))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("TRAINING & VALIDATION LOSS", fontweight="bold", fontsize=11, pad=10)
    ax.legend(loc="upper right")
    ax.set_xticks(list(epochs))


def plot_f1_curve(ax, history):
    epochs = range(1, len(history["val_f1"]) + 1)
    ax.plot(epochs, history["val_f1"],
            color=COLORS["f1"], linewidth=2.5, marker="D",
            markersize=7, zorder=3)
    ax.fill_between(epochs, 0, history["val_f1"],
                    alpha=0.15, color=COLORS["f1"])
    best_f1 = max(history["val_f1"])
    best_idx = history["val_f1"].index(best_f1)
    ax.axhline(y=best_f1, color=COLORS["bullish"], linestyle="--",
               alpha=0.4, linewidth=1)
    ax.annotate(f"Best F1: {best_f1:.4f}",
                xy=(best_idx + 1, best_f1),
                xytext=(10, -20), textcoords="offset points",
                fontsize=8, color=COLORS["bullish"],
                arrowprops=dict(arrowstyle="->", color=COLORS["bullish"], lw=0.8))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("VALIDATION F1 SCORE", fontweight="bold", fontsize=11, pad=10)
    ax.set_xticks(list(epochs))
    ax.set_ylim(0, 1.05)


def plot_confusion_matrix(ax, y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    labels = [LABEL_MAP[i] for i in range(3)]
    cmap = LinearSegmentedColormap.from_list("lz",
        [COLORS["surface"], COLORS["accent"], COLORS["bullish"]])
    sns.heatmap(cm, annot=False, cmap=cmap, ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=1, linecolor=COLORS["grid"],
                cbar_kws={"shrink": 0.8})
    for i in range(3):
        for j in range(3):
            count = cm[i, j]
            pct = cm_normalized[i, j] * 100
            color = "#ffffff" if pct > 50 else COLORS["text"]
            ax.text(j + 0.5, i + 0.5, f"{count}\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color=color)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("CONFUSION MATRIX", fontweight="bold", fontsize=11, pad=10)


def plot_class_metrics(ax, y_true, y_pred):
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
    )
    labels = [LABEL_MAP[i] for i in range(3)]
    x = np.arange(len(labels))
    width = 0.25
    bars_p = ax.bar(x - width, precision, width, label="Precision",
                    color=COLORS["accent"], alpha=0.85, zorder=3)
    bars_r = ax.bar(x, recall, width, label="Recall",
                    color=COLORS["bullish"], alpha=0.85, zorder=3)
    bars_f = ax.bar(x + width, f1, width, label="F1",
                    color=COLORS["neutral"], alpha=0.85, zorder=3)
    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                        f"{height:.2f}", ha="center", va="bottom",
                        fontsize=7, color=COLORS["text"])
    for i, s in enumerate(support):
        ax.text(i, -0.08, f"n={s}", ha="center", fontsize=8,
                color=COLORS["text"], alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("PER-CLASS PRECISION / RECALL / F1", fontweight="bold", fontsize=11, pad=10)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.15)


def plot_confidence_distribution(ax, confidences, predictions):
    class_colors = [COLORS["bearish"], COLORS["neutral"], COLORS["bullish"]]
    class_labels = ["Bearish", "Neutral", "Bullish"]
    for cls in range(3):
        mask = predictions == cls
        if mask.sum() > 0:
            ax.hist(confidences[mask], bins=20, alpha=0.6,
                    color=class_colors[cls], label=class_labels[cls],
                    edgecolor=class_colors[cls], linewidth=0.5, zorder=3)
    ax.axvline(x=0.65, color=COLORS["text"], linestyle="--", alpha=0.5,
               linewidth=1, label="Trade threshold (0.65)")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("PREDICTION CONFIDENCE DISTRIBUTION", fontweight="bold", fontsize=11, pad=10)
    ax.legend(loc="upper left", fontsize=8)


def plot_predictions_table(ax, samples):
    ax.axis("off")
    ax.set_title("SAMPLE PREDICTIONS", fontweight="bold", fontsize=11, pad=10, loc="left")
    if not samples:
        ax.text(0.5, 0.5, "No sample predictions available",
                ha="center", va="center", fontsize=10, color=COLORS["text"])
        return
    col_labels = ["Text", "Pred", "Conf", "Bear", "Neut", "Bull"]
    cell_text = []
    cell_colors = []
    for s in samples[:8]:
        pred = s["prediction"]
        conf = s["confidence"]
        pred_color = {
            "Bearish": COLORS["bearish"],
            "Neutral": COLORS["neutral"],
            "Bullish": COLORS["bullish"],
        }.get(pred, COLORS["text"])
        text = s["text"][:50] + ("..." if len(s["text"]) > 50 else "")
        cell_text.append([
            text, pred, f"{conf:.2%}",
            f"{s['scores']['Bearish']:.3f}",
            f"{s['scores']['Neutral']:.3f}",
            f"{s['scores']['Bullish']:.3f}",
        ])
        cell_colors.append([COLORS["surface"]] * 6)
    table = ax.table(
        cellText=cell_text, colLabels=col_labels, cellColours=cell_colors,
        colColours=[COLORS["grid"]] * 6, loc="center", cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["grid"])
        cell.set_text_props(color=COLORS["text"], fontfamily="monospace")
        if key[0] == 0:
            cell.set_text_props(fontweight="bold", color=COLORS["text"])


def run_evaluation():
    if not ONNX_AVAILABLE:
        print("[EVAL] ONNX/transformers not available — skipping live evaluation")
        return None, None, None, None
    if not os.path.exists(ONNX_PATH):
        print(f"[EVAL] ONNX model not found at {ONNX_PATH} — skipping live evaluation")
        return None, None, None, None
    print("[EVAL] Loading ONNX model for evaluation...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(OUTPUT_DIR)
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(ONNX_PATH, providers=providers)
    try:
        from datasets import load_dataset
        dataset = load_dataset("takala/financial_phrasebank", "sentences_allagree",
                               trust_remote_code=True)["train"]
        from sklearn.model_selection import train_test_split
        sentences = dataset["sentence"]
        labels = dataset["label"]
        _, val_sentences, _, val_labels = train_test_split(
            sentences, labels, test_size=0.2, random_state=42, stratify=labels
        )
    except Exception as e:
        print(f"[EVAL] Could not load dataset: {e}")
        val_sentences = [
            "Revenue surged 45% year-over-year",
            "The company issued a profit warning",
            "Trading volume was average",
            "Strong demand drove record earnings",
            "The SEC launched an investigation",
            "Markets closed flat in quiet trading",
        ]
        val_labels = [2, 0, 1, 2, 0, 1]
    print(f"[EVAL] Running inference on {len(val_sentences)} validation samples...")
    all_preds, all_labels, all_confidences, sample_predictions = [], list(val_labels), [], []
    for text, label in zip(val_sentences, val_labels):
        encoded = tokenizer(text, padding="max_length", truncation=True,
                           max_length=128, return_tensors="np")
        logits = session.run(["logits"], {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        })[0]
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = (exp_l / exp_l.sum(axis=-1, keepdims=True)).squeeze()
        pred = int(np.argmax(probs))
        confidence = float(probs.max())
        all_preds.append(pred)
        all_confidences.append(confidence)
        if len(sample_predictions) < 12:
            sample_predictions.append({
                "text": text, "true_label": LABEL_MAP[label],
                "prediction": LABEL_MAP[pred], "confidence": confidence,
                "correct": pred == label,
                "scores": {
                    "Bearish": float(probs[0]),
                    "Neutral": float(probs[1]),
                    "Bullish": float(probs[2]),
                },
            })
    print(f"[EVAL] Evaluation complete. Accuracy: "
          f"{sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels):.2%}")
    return np.array(all_labels), np.array(all_preds), np.array(all_confidences), sample_predictions


def main():
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument("--no-display", action="store_true", help="Save charts without displaying")
    args = parser.parse_args()
    setup_style()
    os.makedirs(CHARTS_DIR, exist_ok=True)
    meta = load_training_meta()
    config = meta.get("config", {})
    history = meta.get("history", {})
    if not history.get("train_loss"):
        print("[ERROR] No training history found in metadata.")
        sys.exit(1)
    y_true, y_pred, confidences, samples = run_evaluation()

    fig1 = plt.figure(figsize=(16, 12))
    fig1.suptitle("LZ-QUANT  ·  TRAINING RESULTS",
                  fontsize=14, fontweight="bold", color=COLORS["accent"], y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3,
                           left=0.08, right=0.95, top=0.92, bottom=0.08)
    ax1 = fig1.add_subplot(gs[0, 0])
    plot_loss_curves(ax1, history)
    ax2 = fig1.add_subplot(gs[0, 1])
    plot_f1_curve(ax2, history)
    if y_true is not None:
        ax3 = fig1.add_subplot(gs[1, 0])
        plot_confusion_matrix(ax3, y_true, y_pred)
        ax4 = fig1.add_subplot(gs[1, 1])
        plot_class_metrics(ax4, y_true, y_pred)
    else:
        ax3 = fig1.add_subplot(gs[1, :])
        ax3.text(0.5, 0.5, "Install onnxruntime + transformers for confusion matrix & class metrics",
                 ha="center", va="center", fontsize=12, color=COLORS["text"])
        ax3.axis("off")
    chart1_path = os.path.join(CHARTS_DIR, "training_overview.png")
    fig1.savefig(chart1_path)
    print(f"[SAVED] {chart1_path}")

    if y_true is not None:
        fig2 = plt.figure(figsize=(16, 8))
        fig2.suptitle("LZ-QUANT  ·  MODEL CONFIDENCE ANALYSIS",
                      fontsize=14, fontweight="bold", color=COLORS["accent"], y=0.98)
        gs2 = gridspec.GridSpec(1, 2, wspace=0.3, width_ratios=[1, 1.3],
                                left=0.06, right=0.97, top=0.88, bottom=0.1)
        ax5 = fig2.add_subplot(gs2[0, 0])
        plot_confidence_distribution(ax5, confidences, y_pred)
        ax6 = fig2.add_subplot(gs2[0, 1])
        plot_predictions_table(ax6, samples)
        chart2_path = os.path.join(CHARTS_DIR, "confidence_analysis.png")
        fig2.savefig(chart2_path)
        print(f"[SAVED] {chart2_path}")

    fig3, ax7 = plt.subplots(figsize=(10, 5))
    ax7.axis("off")
    fig3.suptitle("LZ-QUANT  ·  TRAINING CONFIGURATION",
                  fontsize=14, fontweight="bold", color=COLORS["accent"], y=0.95)
    config_text = (
        f"Model:          {config.get('model_name', 'distilbert-base-uncased')}\n"
        f"LoRA Rank:      {config.get('lora_r', 8)}  (alpha: {config.get('lora_alpha', 16)})\n"
        f"Target Modules: {config.get('lora_target_modules', ['q_lin', 'v_lin'])}\n"
        f"Batch Size:     {config.get('batch_size', 8)} × {config.get('gradient_accumulation_steps', 4)} "
        f"= {config.get('batch_size', 8) * config.get('gradient_accumulation_steps', 4)} effective\n"
        f"Learning Rate:  {config.get('learning_rate', 2e-4)}\n"
        f"Epochs:         {config.get('num_epochs', 5)}\n"
        f"Seq Length:     {config.get('max_seq_length', 128)}\n"
        f"FP16:           {config.get('fp16', True)}\n"
        f"Best Epoch:     {meta.get('best_epoch', '?')}\n"
        f"Best Val Loss:  {min(history['val_loss']):.4f}\n"
        f"Best Val F1:    {max(history['val_f1']):.4f}\n"
    )
    ax7.text(0.05, 0.5, config_text, transform=ax7.transAxes,
             fontsize=12, fontfamily="monospace", color=COLORS["text"],
             verticalalignment="center",
             bbox=dict(boxstyle="round,pad=0.8", facecolor=COLORS["surface"],
                       edgecolor=COLORS["grid"]))
    chart3_path = os.path.join(CHARTS_DIR, "training_config.png")
    fig3.savefig(chart3_path)
    print(f"[SAVED] {chart3_path}")

    print(f"\n{'─' * 60}")
    print(f"  All charts saved to: {CHARTS_DIR}/")
    print(f"  Files:")
    for f in sorted(os.listdir(CHARTS_DIR)):
        size = os.path.getsize(os.path.join(CHARTS_DIR, f)) / 1024
        print(f"    {f:<35} {size:.0f} KB")
    print(f"{'─' * 60}")

    if not args.no_display:
        print(f"\n  Displaying charts... (close windows to exit)")
        plt.show()
    else:
        print(f"\n  Charts saved (--no-display mode)")

if __name__ == "__main__":
    main()