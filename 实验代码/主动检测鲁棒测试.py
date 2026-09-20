import os
import json
import math
import random
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

ATTACK_RATES = [0.0, 0.1, 0.2, 0.3]  # Attack intensity: 0%, 10%, 20%, 30% word replacement

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

# ==================== 2. KGW Z-Score Computation ====================
def calculate_kgw_z_score(text, tokenizer, real_vocab_size, gamma=0.25, hash_key=42):
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

# ==================== 3. Simulated Attack Helper ====================
def apply_random_replacement(text, replacement_rate):
    """Simulates text editing attack by randomly replacing characters/words."""
    if replacement_rate <= 0.0 or not text:
        return text

    chars = list(text)
    num_to_replace = int(len(chars) * replacement_rate)
    indices_to_replace = random.sample(range(len(chars)), num_to_replace)

    # Replace with random common punctuation or placeholders
    replacement_pool = ['，', '的', '是', '在', ' ', '。']
    for idx in indices_to_replace:
        chars[idx] = random.choice(replacement_pool)

    return "".join(chars)

# ==================== 4. Plotting Function ====================
def plot_robustness_curves(robustness_results):
    os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    markers = ['o', 's', '^']
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C']

    rates_pct = [int(r * 100) for r in ATTACK_RATES]

    for idx, (model_name, tpr_list) in enumerate(robustness_results.items()):
        ax.plot(rates_pct, tpr_list, label=model_name, marker=markers[idx], 
                color=colors[idx], linewidth=2.2, markersize=7)

    ax.set_xlabel('Substitution Attack Intensity (%)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Watermark Detection Rate (TPR %)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'Watermark Robustness under Random Substitution Attack ($\\tau = {Z_THRESHOLD}$)', 
                 fontsize=13, fontweight='bold', pad=15)
    
    ax.set_xticks(rates_pct)
    ax.set_xticklabels([f'{r}%' for r in rates_pct], fontsize=10, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', frameon=True)

    plt.tight_layout()
    save_path = os.path.join(FIGURE_SAVE_DIR, "watermark_robustness_attack.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n📊 Robustness Line Chart saved successfully: {save_path}")

# ==================== 5. Main Evaluation Flow ====================
def evaluate_robustness():
    print("=" * 75)
    print(f"Execution Environment: [{DEVICE.upper()}] | Watermark Robustness Evaluation")
    print("=" * 75)

    wm_dir = os.path.join(OUTPUT_DIR, "watermarked_baseline")
    if not os.path.exists(wm_dir):
        print("❌ Watermarked baseline directory not found.")
        return

    robustness_results = {}

    for model_name, cfg in MODEL_CONFIGS.items():
        tokenizer_path = cfg["tokenizer_path"]
        model_key = cfg["key"]

        if not os.path.exists(tokenizer_path):
            continue

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        try:
            config = AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True)
            real_vocab_size = getattr(config, "vocab_size", tokenizer.vocab_size)
        except Exception:
            real_vocab_size = tokenizer.vocab_size

        model_tag = model_name.lower().replace(".", "").replace("-", "_")
        wm_file = next((os.path.join(wm_dir, f) for f in os.listdir(wm_dir) if model_tag in f), None)

        if not wm_file:
            continue

        with open(wm_file, "r", encoding="utf-8") as f:
            wm_data = json.load(f)

        print(f"\n🔍 Evaluating Robustness for [{model_name}]...")
        tpr_by_rate = []

        for rate in ATTACK_RATES:
            random.seed(42)  # Fixed seed for reproducible attack simulation
            tp_count = 0

            for item in wm_data:
                original_text = item.get("AI生成文本", item.get("文本", ""))
                
                # Apply attack
                attacked_text = apply_random_replacement(original_text, rate)
                
                z, _ = calculate_kgw_z_score(
                    text=attacked_text,
                    tokenizer=tokenizer,
                    real_vocab_size=real_vocab_size,
                    gamma=KGW_GAMMA,
                    hash_key=model_key
                )
                if z >= Z_THRESHOLD:
                    tp_count += 1

            tpr = (tp_count / len(wm_data)) * 100 if wm_data else 0.0
            tpr_by_rate.append(tpr)
            print(f"  ├─ Attack Intensity {int(rate*100)}%: TPR = {tpr:.2f}% ({tp_count}/{len(wm_data)})")

        robustness_results[model_name] = tpr_by_rate

    if robustness_results:
        plot_robustness_curves(robustness_results)

if __name__ == "__main__":
    evaluate_robustness()