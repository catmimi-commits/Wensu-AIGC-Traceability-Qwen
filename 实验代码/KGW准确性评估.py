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
Z_THRESHOLD = 3.5       # Decision Threshold (\tau = 3.5)

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

# Set global Matplotlib params for academic paper presentation
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
def plot_z_score_distribution(results_data):
    """Generates Box Plots for Z-Score distributions across models."""
    os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    models = list(results_data.keys())
    x = np.arange(len(models))
    width = 0.35

    unwm_scores = [results_data[m]["unwm_z"] for m in models]
    wm_scores = [results_data[m]["wm_z"] for m in models]

    # Plot Boxplots
    bp1 = ax.boxplot(unwm_scores, positions=x - width/2, widths=0.28, patch_artist=True,
                     boxprops=dict(facecolor='#A6CEE3', color='#1F78B4'),
                     medianprops=dict(color='black', linewidth=1.5))
    bp2 = ax.boxplot(wm_scores, positions=x + width/2, widths=0.28, patch_artist=True,
                     boxprops=dict(facecolor='#FB9A99', color='#E31A1C'),
                     medianprops=dict(color='black', linewidth=1.5))

    # Add Decision Threshold Line
    ax.axhline(y=Z_THRESHOLD, color='#333333', linestyle='--', linewidth=1.5, 
               label=f'Threshold ($\\tau = {Z_THRESHOLD}$)')

    # Labels and Titles
    ax.set_ylabel('Z-score Value', fontsize=12, fontweight='bold')
    ax.set_title('Z-score Distributions: Unwatermarked vs. Watermarked Text', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Legend
    ax.legend([bp1["boxes"][0], bp2["boxes"][0], ax.get_lines()[0]], 
              ['Unwatermarked', 'Watermarked', f'Threshold ($\\tau = {Z_THRESHOLD}$)'], 
              loc='upper left', frameon=True)

    plt.tight_layout()
    save_path = os.path.join(FIGURE_SAVE_DIR, "z_score_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  📊 Chart saved successfully: {save_path}")

def plot_tpr_fpr_metrics(results_data):
    """Generates Bar Charts for TPR and FPR performance."""
    os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    models = list(results_data.keys())
    x = np.arange(len(models))
    width = 0.35

    tprs = [results_data[m]["tpr"] for m in models]
    fprs = [results_data[m]["fpr"] for m in models]

    rects1 = ax.bar(x - width/2, tprs, width, label='TPR (True Positive Rate)', color='#2CA02C', alpha=0.85)
    rects2 = ax.bar(x + width/2, fprs, width, label='FPR (False Positive Rate)', color='#D62728', alpha=0.85)

    # Add values above bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Watermark Detection Metrics (Threshold $\\tau = {Z_THRESHOLD}$)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    save_path = os.path.join(FIGURE_SAVE_DIR, "tpr_fpr_metrics.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  📊 Chart saved successfully: {save_path}")

# ==================== 4. Main Evaluation Flow ====================
def evaluate_watermark_detection():
    print("=" * 70)
    print(f"Execution Environment: [{DEVICE.upper()}] | Decision Threshold: [{Z_THRESHOLD}]")
    print("=" * 70)

    unwm_dir = os.path.join(OUTPUT_DIR, "unwatermarked")
    wm_dir = os.path.join(OUTPUT_DIR, "watermarked_baseline")

    if not os.path.exists(unwm_dir) or not os.path.exists(wm_dir):
        print("❌ Dataset directory not found. Please check paths.")
        return

    results_data = {}

    for model_name, cfg in MODEL_CONFIGS.items():
        tokenizer_path = cfg["tokenizer_path"]
        model_key = cfg["key"]
        
        print(f"\n🔍 Evaluating Model: [{model_name}] (Secret Key: {model_key})")
        
        if not os.path.exists(tokenizer_path):
            print(f"  ⚠️ Tokenizer path does not exist, skipping: {tokenizer_path}")
            continue

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        try:
            config = AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True)
            real_vocab_size = getattr(config, "vocab_size", tokenizer.vocab_size)
        except Exception:
            real_vocab_size = tokenizer.vocab_size

        model_tag = model_name.lower().replace(".", "").replace("-", "_")
        unwm_file = next((os.path.join(unwm_dir, f) for f in os.listdir(unwm_dir) if model_tag in f), None)
        wm_file = next((os.path.join(wm_dir, f) for f in os.listdir(wm_dir) if model_tag in f), None)

        if not unwm_file or not wm_file:
            print(f"  ⚠️ Data files for [{model_tag}] not found, skipping")
            continue

        with open(unwm_file, "r", encoding="utf-8") as f:
            unwm_data = json.load(f)
        with open(wm_file, "r", encoding="utf-8") as f:
            wm_data = json.load(f)

        # 1. Evaluate Unwatermarked Group
        unwm_fp_count = 0
        unwm_z_scores = []
        for item in unwm_data:
            text = item.get("AI生成文本", item.get("文本", ""))
            z, _ = calculate_kgw_z_score(text, tokenizer, real_vocab_size, gamma=KGW_GAMMA, hash_key=model_key)
            unwm_z_scores.append(z)
            if z >= Z_THRESHOLD:
                unwm_fp_count += 1

        # 2. Evaluate Watermarked Group
        wm_tp_count = 0
        wm_z_scores = []
        for item in wm_data:
            text = item.get("AI生成文本", item.get("文本", ""))
            z, _ = calculate_kgw_z_score(text, tokenizer, real_vocab_size, gamma=KGW_GAMMA, hash_key=model_key)
            wm_z_scores.append(z)
            if z >= Z_THRESHOLD:
                wm_tp_count += 1

        total_unwm = len(unwm_data)
        total_wm = len(wm_data)
        
        fpr = (unwm_fp_count / total_unwm) * 100 if total_unwm > 0 else 0.0
        tpr = (wm_tp_count / total_wm) * 100 if total_wm > 0 else 0.0

        print(f"  ├─ [Unwatermarked] Samples: {total_unwm} | Avg Z-score: {sum(unwm_z_scores)/len(unwm_z_scores):.2f}")
        print(f"  ├─ [Watermarked]   Samples: {total_wm} | Avg Z-score: {sum(wm_z_scores)/len(wm_z_scores):.2f}")
        print(f"  └─ 📊 FPR: {fpr:.2f}% | TPR: {tpr:.2f}%")

        # Record results for plotting
        results_data[model_name] = {
            "unwm_z": unwm_z_scores,
            "wm_z": wm_z_scores,
            "tpr": tpr,
            "fpr": fpr
        }

    # Generate charts
    if results_data:
        print("\n📈 Rendering academic charts...")
        plot_z_score_distribution(results_data)
        plot_tpr_fpr_metrics(results_data)

if __name__ == "__main__":
    evaluate_watermark_detection()