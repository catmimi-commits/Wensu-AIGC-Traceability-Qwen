import json
import os
import re
import joblib
import jieba
import jieba.posseg as pseg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# ==================== 1. 还原训练代码中的特征提取函数 ====================
def extract_length_invariant_features(text):
    """
    12维基础特征 + N-gram特征，彻底强化Qwen模型区分度 (原训练脚本提取函数)
    """
    if not isinstance(text, str) or text.strip() == "":
        return [0.0] * 20  # 扩展到20维特征
    
    text = text.strip()
    char_count = len(text)
    
    # 只分词一次（带词性），避免重复计算
    words_with_pos = list(pseg.cut(text))
    words = [w for w, pos in words_with_pos]
    word_count = len(words) if words else 1  # 避免除0
    
    features = []
    
    # --- 1. 原有12维基础特征 ---
    avg_word_len = char_count / word_count
    features.append(avg_word_len / 6.0)
    
    punct = re.findall(r'[，。！？；：“”‘’（）【】、]', text)
    features.append(len(punct)/char_count if char_count else 0.0)
    
    digits = re.findall(r'\d+', text)
    digit_total = len(''.join(digits))
    features.append(digit_total/char_count if char_count else 0.0)
    
    stopwords = {'的','了','在','是','和','与','有','为','以','而','就','都','这','那','使','由'}
    stop_cnt = sum(1 for w in words if w in stopwords)
    features.append(stop_cnt / word_count)
    
    features.append(len(set(words)) / word_count)
    
    sentences = re.split(r'[。！？]', text)
    valid_sents = [s.strip() for s in sentences if s.strip()]
    sent_word_counts = [len(jieba.lcut(s)) for s in valid_sents] if valid_sents else []
    avg_sent_word = np.mean(sent_word_counts) if sent_word_counts else 0
    features.append(avg_sent_word / word_count if word_count else 0)
    
    content_pos = {'a','v','n','nr','ns','nt','nz','vd','vn'}
    content_cnt = sum(1 for w,p in words_with_pos if p in content_pos)
    features.append(1 - content_cnt/word_count)
    
    long_cnt = sum(1 for c in sent_word_counts if c > 18)
    features.append(long_cnt/len(valid_sents) if valid_sents else 0)
    
    summary_words = {'建议','本文','呼吁','总结','综上所述','总之','因此','由此可见'}
    sm_cnt = sum(1 for w in words if w in summary_words)
    features.append(sm_cnt / word_count)
    
    adv_words = {'很','更','非常','十分','极其','格外','稍微','几乎','往往','常常','大致','通常'}
    adv_cnt = sum(1 for w in words if w in adv_words)
    features.append(adv_cnt / word_count)
    
    conj_words = {'因为','所以','但是','然而','而且','此外','同时','于是','据此','对此'}
    conj_cnt = sum(1 for w in words if w in conj_words)
    features.append(conj_cnt / word_count)
    
    short_func = sum(1 for w in words if len(w) == 1 and w in {'会','能','可','要','让','使','与','或','即'})
    features.append(short_func / word_count)

    # --- 2. 新增8维N-gram特征 ---
    qwen25_ngrams = {'，这', '，而', '，但', '，更', '，非常', '，显著'}
    qwen2_ngrams = {'，本文', '，建议', '，此外', '，同时', '，因此', '，综上所述'}
    
    text_ngrams = set()
    for i in range(len(text) - 1):
        text_ngrams.add(text[i] + text[i+1])
    
    qwen25_cnt = sum(1 for ng in qwen25_ngrams if ng in text_ngrams)
    features.append(qwen25_cnt / len(text) if text else 0)
    
    qwen2_cnt = sum(1 for ng in qwen2_ngrams if ng in text_ngrams)
    features.append(qwen2_cnt / len(text) if text else 0)
    
    common_ngrams = ['的', '了', '在', '是', '和', '与']
    for c in common_ngrams:
        features.append(text.count(c) / len(text) if text else 0)

    return features

# ==================== 2. 模型加载配置 ====================
MODEL_DIR = r"C:\Users\29276\Desktop\大创与竞赛\大创实验资料\model"
MODEL_PATH = os.path.join(MODEL_DIR, 'ai_detector_tracking.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder_tracking.pkl')
ATTACK_DATASET_PATH = r"C:\Users\29276\Desktop\大创与竞赛\大创实验资料\被动鲁棒性.json"

print("Loading AI tracking model and encoder...")
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# ==================== 3. 读取数据与评测逻辑 ====================
print(f"Loading robustness dataset: {ATTACK_DATASET_PATH} ...")
with open(ATTACK_DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

def evaluate_group(text_list, true_labels):
    """批量提取20维特征，并调用模型预测"""
    X_features = [extract_length_invariant_features(t) for t in text_list]
    X_features = np.array(X_features)
    
    pred_codes = model.predict(X_features)
    preds = label_encoder.inverse_transform(pred_codes)
    
    acc = accuracy_score(true_labels, preds)
    return acc

# ==================== 4. 分组评测与数据收集 ====================
results_summary = []

# A. Baseline 测试
baseline_df = df.drop_duplicates(subset=["新闻序号", "真实模型标签"]).copy()
base_acc = evaluate_group(baseline_df["原始AI文本"].tolist(), baseline_df["真实模型标签"].tolist())

results_summary.append({
    "Attack_Type": "Baseline",
    "Display_Label": "Baseline\n(Original)",
    "Accuracy": round(base_acc * 100, 2),
    "Drop_Rate": 0.00
})

# B. 各攻击等级测试
attack_types = df["攻击类型"].unique()

# 定义攻击标签的英文显示映射
label_mapping = {
    "L1_Light_Polish": "L1: Light\nPolish",
    "L2_Structure_Rewrite": "L2: Structure\nRewrite",
    "L3_Paraphrase_Stylize": "L3: Heavy\nParaphrase"
}

for attack_type in sorted(attack_types):
    group_df = df[df["攻击类型"] == attack_type]
    acc = evaluate_group(group_df["攻击改写文本"].tolist(), group_df["真实模型标签"].tolist())
    
    drop_pct = ((base_acc - acc) / base_acc) * 100
    display_label = label_mapping.get(attack_type, attack_type)
    
    results_summary.append({
        "Attack_Type": attack_type,
        "Display_Label": display_label,
        "Accuracy": round(acc * 100, 2),
        "Drop_Rate": round(drop_pct, 2)
    })

plot_df = pd.DataFrame(results_summary)

print("\n" + "="*50)
print("  Passive Detection Robustness Evaluation Summary  ")
print("="*50)
print(plot_df[["Attack_Type", "Accuracy", "Drop_Rate"]].to_string(index=False))

# ==================== 5. 生成英文学术图表 (JPG) ====================
print("\nGenerating publication-quality English plot...")

# 设置学术论文美化风格
sns.set_theme(style="whitegrid", font="sans-serif")
plt.figure(figsize=(8, 5.5), dpi=300)

# 定义调色盘 (优雅蓝赤配色)
bar_color = "#4C72B0"
line_color = "#C44E52"

# 绘制柱状图 (Classification Accuracy)
ax1 = plt.gca()
bars = ax1.bar(
    plot_df["Display_Label"], 
    plot_df["Accuracy"], 
    color=bar_color, 
    width=0.45, 
    alpha=0.85, 
    edgecolor="black",
    linewidth=1.2,
    label="Accuracy (%)"
)

# 绘制折线图 (Accuracy Drop)
ax1.plot(
    plot_df["Display_Label"], 
    plot_df["Accuracy"], 
    color=line_color, 
    marker="o", 
    linewidth=2.5, 
    markersize=8, 
    linestyle="--",
    label="Trend Line"
)

# 在柱子上标注具体数值 (%)
for bar in bars:
    yval = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0, 
        yval + 1.5, 
        f"{yval:.2f}%", 
        ha="center", 
        va="bottom", 
        fontsize=10, 
        fontweight="bold", 
        color="#222222"
    )

# 标注三分类的随机猜测基准线 (33.33%)
ax1.axhline(y=33.33, color="gray", linestyle=":", linewidth=1.5, label="Random Guess Baseline (33.33%)")

# 坐标轴细节设置
ax1.set_title("Robustness Decay of Passive Stylistic Classifier\nunder Text Paraphrasing Attacks", fontsize=13, fontweight="bold", pad=15)
ax1.set_xlabel("Attack Type / Paraphrasing Intensity", fontsize=11, fontweight="bold", labelpad=10)
ax1.set_ylabel("Classification Accuracy (%)", fontsize=11, fontweight="bold", labelpad=10)
ax1.set_ylim(0, 105)

# 优化网格与边框
ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# 调整 Legend 位置
ax1.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)

plt.tight_layout()

# 保存为 JPG 格式图片
output_jpg_path = "passive_detection_robustness_english.jpg"
plt.savefig(output_jpg_path, format="jpg", dpi=300, bbox_inches="tight")
plt.close()

print(f"Chart successfully saved as high-resolution image: {output_jpg_path}")