import os
import json
import math
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoConfig

# ==================== 1. Global Configurations ====================
OUTPUT_DIR = r"C:\Users\29276\Desktop\大创与竞赛\大创实验资料\主动检测dataset"
FIGURE_SAVE_DIR = r"C:\Users\29276\Desktop\大创与竞赛\大创实验资料\figures"

KGW_GAMMA = 0.25        # Green List ratio
Z_THRESHOLD = 3.5       # Decision Threshold

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIGS = {
    "Qwen1.5-0.5B": {
        "tokenizer_path": r"C:\Users\29276\Desktop\大创与竞赛\本地模型库\Qwen1.5-0.5B\Qwen1.5-0.5B-Instruct",
        "key": 62
    },
    "Qwen2-0.5B": {
        "tokenizer_path": r"C:\Users\29276\Desktop\大创与竞赛\本地模型库\Qwen2-0.5B-Instruct",
        "key": 52
    },
    "Qwen2.5-1.5B": {
        "tokenizer_path": r"C:\Users\29276\Desktop\大创与竞赛\本地模型库\Qwen2.5-1.5b\Qwen2.5-1.5B-Instruct",
        "key": 42
    }
}

# Set global Matplotlib params for academic visualization
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

# ==================== 2. KGW Z-Score Computation ====================
def calculate_kgw_z_score(text, tokenizer, real_vocab_size, gamma=0.25, hash_key=42):
    """Computes KGW Z-score strictly aligned with generation-side LogitsProcessor."""
    if not text or not text.strip():
        return 0.0, 0

    tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(tokens)
    
    if total_tokens < 5:
        return 0.0, total_tokens

    green_count = 0
    greenlist_size = int(real_vocab_size * gamma)

    for i in range(1, total_tokens):
        prev_token = tokens[i - 1]
        curr_token = tokens[i]

        rng = torch.Generator(device=DEVICE)
        rng.manual_seed((hash_key * 10007 + prev_token) % (2**32 - 1))
        
        permutation = torch.randperm(real_vocab_size, generator=rng, device=DEVICE)
        green_tokens = set(permutation[:greenlist_size].tolist())

        if curr_token in green_tokens:
            green_count += 1

    N = total_tokens - 1
    if N <= 0:
        return 0.0, total_tokens

    expected_green = gamma * N
    std_dev = math.sqrt(N * gamma * (1 - gamma))
    z_score = (green_count - expected_green) / std_dev if std_dev > 0 else 0.0
    
    return z_score, total_tokens

# ==================== 3. Plotting Functions ====================
def plot_confusion_matrix(conf_matrix, model_names):
    """Generates normalized confusion matrix heatmap for source attribution."""
    os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)

    # Normalize matrix by row (True Source)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.divide(conf_matrix, row_sums, out=np.zeros_like(conf_matrix, dtype=float), where=row_sums!=0) * 100

    im = ax.imshow(norm_matrix, cmap='Blues', vmin=0, vmax=100)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Attribution Accuracy (%)', rotation=-90, va="bottom", fontsize=11, fontweight='bold')

    # Show ticks and labels
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=10, fontweight='bold')
    ax.set_yticklabels(model_names, fontsize=10, fontweight='bold')

    ax.set_xlabel('Predicted Source (Detected Key)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Source (Actual Key)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Watermark Source Attribution Confusion Matrix', fontsize=13, fontweight='bold', pad=15)

    # Annotate values inside cells
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            count = int(conf_matrix[i, j])
            pct = norm_matrix[i, j]
            text_color = "white" if pct > 50 else "black"
            ax.text(j, i, f"{pct:.1f}%\n({count})", ha="center", va="center", color=text_color, fontweight='bold', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(FIGURE_SAVE_DIR, "source_attribution_confusion_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  📊 Confusion Matrix saved successfully: {save_path}")

def plot_cross_key_z_matrix(z_response_matrix, model_names):
    """Generates a heatmap showing average Z-scores when testing texts against all candidate keys."""
    os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    im = ax.imshow(z_response_matrix, cmap='YlOrRd', vmin=0)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Average Z-score', rotation=-90, va="bottom", fontsize=11, fontweight='bold')

    ax.set_xticks(np.arange(len(model_names)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=10, fontweight='bold')
    ax.set_yticklabels(model_names, fontsize=10, fontweight='bold')

    ax.set_xlabel('Candidate Decoder (Key Tested)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Source Model', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Cross-Key Z-Score Response Matrix (Key Orthogonality)', fontsize=13, fontweight='bold', pad=15)

    for i in range(len(model_names)):
        for j in range(len(model_names)):
            val = z_response_matrix[i, j]
            text_color = "white" if val > (z_response_matrix.max() / 2) else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight='bold', fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(FIGURE_SAVE_DIR, "cross_key_z_score_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  📊 Cross-Key Z-score Matrix saved successfully: {save_path}")

# ==================== 4. Main Source Attribution Evaluation ====================
def evaluate_source_attribution():
    print("=" * 75)
    print(f"Execution Environment: [{DEVICE.upper()}] | Source Attribution Analysis")
    print("=" * 75)

    wm_dir = os.path.join(OUTPUT_DIR, "watermarked_baseline")

    if not os.path.exists(wm_dir):
        print("❌ Watermarked baseline dataset directory not found.")
        return

    # Pre-load all tokenizers and config dimensions
    model_names = list(MODEL_CONFIGS.keys())
    tokenizers = {}
    vocab_sizes = {}

    print("🔄 Pre-loading candidate tokenizers and configurations...")
    for m_name in model_names:
        t_path = MODEL_CONFIGS[m_name]["tokenizer_path"]
        if os.path.exists(t_path):
            tokenizers[m_name] = AutoTokenizer.from_pretrained(t_path, trust_remote_code=True)
            try:
                cfg = AutoConfig.from_pretrained(t_path, trust_remote_code=True)
                vocab_sizes[m_name] = getattr(cfg, "vocab_size", tokenizers[m_name].vocab_size)
            except Exception:
                vocab_sizes[m_name] = tokenizers[m_name].vocab_size
        else:
            print(f"⚠️ Tokenizer not found for {m_name} at {t_path}")
            return

    num_models = len(model_names)
    conf_matrix = np.zeros((num_models, num_models), dtype=int)
    z_response_matrix = np.zeros((num_models, num_models), dtype=float)

    total_samples_evaluated = 0
    correct_attribution_count = 0

    # Cross-evaluating every watermarked file against all candidate keys
    for true_idx, true_model in enumerate(model_names):
        model_tag = true_model.lower().replace(".", "").replace("-", "_")
        wm_file = next((os.path.join(wm_dir, f) for f in os.listdir(wm_dir) if model_tag in f), None)

        if not wm_file:
            print(f"⚠️ Data file for [{true_model}] not found, skipping.")
            continue

        with open(wm_file, "r", encoding="utf-8") as f:
            wm_data = json.load(f)

        print(f"\n🔍 Testing Attribution for Source: [{true_model}] (True Key: {MODEL_CONFIGS[true_model]['key']})")

        sample_z_scores = {cand_idx: [] for cand_idx in range(num_models)}

        for item in wm_data:
            text = item.get("AI生成文本", item.get("文本", ""))
            if not text.strip():
                continue

            cand_z_scores = {}
            for cand_idx, cand_model in enumerate(model_names):
                cand_key = MODEL_CONFIGS[cand_model]["key"]
                z, _ = calculate_kgw_z_score(
                    text=text,
                    tokenizer=tokenizers[cand_model],
                    real_vocab_size=vocab_sizes[cand_model],
                    gamma=KGW_GAMMA,
                    hash_key=cand_key
                )
                cand_z_scores[cand_idx] = z
                sample_z_scores[cand_idx].append(z)

            # Assign prediction to key with maximum Z-score
            pred_idx = max(cand_z_scores, key=cand_z_scores.get)
            max_z = cand_z_scores[pred_idx]

            # If max Z-score is below threshold, it's considered unidentifiable (optional safeguard)
            if max_z < Z_THRESHOLD:
                pred_idx = -1  # Unattributed / Unknown

            if pred_idx == true_idx:
                correct_attribution_count += 1
                conf_matrix[true_idx, true_idx] += 1
            elif pred_idx != -1:
                conf_matrix[true_idx, pred_idx] += 1

            total_samples_evaluated += 1

        # Calculate average Z-scores across keys for current source model
        for cand_idx in range(num_models):
            avg_z = sum(sample_z_scores[cand_idx]) / len(sample_z_scores[cand_idx]) if sample_z_scores[cand_idx] else 0.0
            z_response_matrix[true_idx, cand_idx] = avg_z

        true_acc = (conf_matrix[true_idx, true_idx] / len(wm_data)) * 100 if wm_data else 0.0
        print(f"  └─ True Key Avg Z-score: {z_response_matrix[true_idx, true_idx]:.2f}")
        print(f"  └─ Source Attribution Accuracy for [{true_model}]: {true_acc:.2f}%")

    overall_accuracy = (correct_attribution_count / total_samples_evaluated) * 100 if total_samples_evaluated > 0 else 0.0
    print("\n" + "=" * 75)
    print(f"🎯 Overall Source Attribution Accuracy: {overall_accuracy:.2f}% ({correct_attribution_count}/{total_samples_evaluated})")
    print("=" * 75)

    # Plot Visualizations
    print("\n📈 Generating Attribution Analysis Figures...")
    plot_confusion_matrix(conf_matrix, model_names)
    plot_cross_key_z_matrix(z_response_matrix, model_names)

if __name__ == "__main__":
    evaluate_source_attribution()