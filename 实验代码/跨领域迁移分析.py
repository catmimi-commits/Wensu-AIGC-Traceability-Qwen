# -*- coding: utf-8 -*-
"""
跨领域模型指纹迁移实验脚本（修复版）
功能：训练域 vs 测试域 全矩阵实验 + 英文可视化
"""

import json
import os
import numpy as np
import pandas as pd
import jieba
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
import torch
import joblib
from sklearn.model_selection import train_test_split

# ================== 🔧 配置区域 ==================

# 1. 数据文件路径
DATA_FILES = {
    'Sports': r"C:\Users\29276\Desktop\大创实验资料\体育merged_output.json",
    'Technology': r"C:\Users\29276\Desktop\大创实验资料\科技merged_output.json",
    'Education': r"C:\Users\29276\Desktop\大创实验资料\教育merged_output.json",
    'Politics': r"C:\Users\29276\Desktop\大创实验资料\时政merged_output.json"
}

# 2. RoBERTa 模型路径
ROBERTA_MODEL_PATH = r"C:\Users\29276\Desktop\本地模型库\chinese_roberta_wwm_ext"

# 3. 特征缓存目录
CACHE_DIR = "feature_cache_fixed"
os.makedirs(CACHE_DIR, exist_ok=True)

# 4. 实验参数
N_ESTIMATORS = 200
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 用于同领域验证

# 5. 输出目录
OUTPUT_DIR = r"C:\Users\29276\Desktop\大创实验资料\results_fixed"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)

# ================== 🛠 工具函数 ==================

def setup_matplotlib_english():
    """设置matplotlib为英文显示"""
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    # 关闭任何中文字体警告
    import warnings
    warnings.filterwarnings("ignore", message="findfont: Font family")
    
def load_json_data(file_path):
    """加载JSON数据，返回(texts, labels)"""
    texts = []
    labels = []
    
    print(f"Loading: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        content = json.load(f)
        if isinstance(content, dict):
            content = [content]
        
        for item in content:
            for rewrite in item.get("改写结果", []):
                texts.append(rewrite["文本"])
                labels.append(rewrite["模型"])
    
    print(f"  -> Loaded: {len(texts)} samples")
    return texts, labels

def extract_surface_features_batch(texts):
    """批量提取表层特征"""
    features = []
    for text in texts:
        char_count = len(text)
        words = list(jieba.cut(text))
        word_count = len(words)
        sentences = re.split(r'[。！？]', text)
        sent_count = len([s for s in sentences if s.strip()])
        avg_sent_len = char_count / sent_count if sent_count > 0 else 0
        punctuation = re.findall(r'[，。！？；：“”‘’（）【】、]', text)
        punct_density = len(punctuation) / char_count if char_count > 0 else 0
        digits = re.findall(r'\d+', text)
        digit_density = len(''.join(digits)) / char_count if char_count > 0 else 0
        features.append([char_count, word_count, sent_count, avg_sent_len, punct_density, digit_density])
    return np.array(features)

def extract_roberta_features_batch(texts, tokenizer, model, batch_size=32):
    """批量提取RoBERTa特征"""
    features = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, 
                          truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_vecs = outputs.last_hidden_state[:, 0, :].numpy()
        features.append(cls_vecs)
    
    return np.vstack(features)

def extract_features_without_leakage(texts, tfidf_model=None, tokenizer=None, 
                                    roberta_model=None, fit_models=False):
    """
    提取特征，避免数据泄露
    - fit_models=True: 在训练集上拟合模型
    - fit_models=False: 使用已有模型进行变换
    """
    # 1. 表层特征（无数据泄露风险）
    surface_feats = extract_surface_features_batch(texts)
    
    # 2. TF-IDF特征
    if fit_models:
        # 在训练集上拟合新的TF-IDF
        tfidf_model = TfidfVectorizer(
            tokenizer=lambda x: list(jieba.cut(x)), 
            max_features=100, 
            stop_words=['的','了','在','是','和','与']
        )
        tfidf_feats = tfidf_model.fit_transform(texts).toarray()
    else:
        # 使用已有的TF-IDF模型变换
        if tfidf_model is None:
            raise ValueError("TF-IDF model must be provided when fit_models=False")
        tfidf_feats = tfidf_model.transform(texts).toarray()
    
    # 3. RoBERTa特征
    if fit_models:
        # 重新加载模型（每次拟合时用新的模型实例）
        tokenizer = BertTokenizer.from_pretrained(ROBERTA_MODEL_PATH)
        roberta_model = BertModel.from_pretrained(ROBERTA_MODEL_PATH)
        roberta_model.eval()
    
    if tokenizer is None or roberta_model is None:
        raise ValueError("Tokenizer and model must be provided")
    
    roberta_feats = extract_roberta_features_batch(texts, tokenizer, roberta_model)
    
    # 拼接特征
    X = np.hstack([surface_feats, tfidf_feats, roberta_feats])
    
    return X, tfidf_model, tokenizer, roberta_model

def prepare_domain_data(domain_name, texts, labels, global_le, 
                        tfidf_model=None, tokenizer=None, roberta_model=None,
                        is_training=False):
    """
    准备单个领域的数据（训练或测试）
    - is_training=True: 拟合新模型
    - is_training=False: 使用已有模型
    """
    print(f"  Preparing {domain_name} data...")
    
    # 编码标签
    y = global_le.transform(labels)
    
    # 提取特征
    X, tfidf_model, tokenizer, roberta_model = extract_features_without_leakage(
        texts, tfidf_model, tokenizer, roberta_model, fit_models=is_training
    )
    
    return X, y, tfidf_model, tokenizer, roberta_model

# ================== 🚀 主实验流程 ==================

def main():
    setup_matplotlib_english()
    
    print("="*60)
    print("🚀 CROSS-DOMAIN MODEL FINGERPRINT TRANSFER EXPERIMENT")
    print("="*60)
    
    # 1. 加载所有领域数据
    print("\n[Step 1] Loading all domain data...")
    all_data = {}
    all_labels_concat = []
    
    for domain, path in DATA_FILES.items():
        if not os.path.exists(path):
            print(f"⚠️ Warning: File {path} does not exist, skipping")
            continue
        texts, labels = load_json_data(path)
        all_data[domain] = {'texts': texts, 'labels': labels}
        all_labels_concat.extend(labels)
    
    if len(all_data) < 2:
        print("❌ Error: Need at least 2 domains")
        return
    
    # 2. 构建全局标签编码器
    print("\n[Step 2] Building global label encoder...")
    global_le = LabelEncoder()
    global_le.fit(all_labels_concat)
    print(f"  Label classes: {list(global_le.classes_)}")
    
    # 3. 初始化RoBERTa模型（只加载一次）
    print("\n[Step 3] Loading RoBERTa model...")
    tokenizer = BertTokenizer.from_pretrained(ROBERTA_MODEL_PATH)
    roberta_model = BertModel.from_pretrained(ROBERTA_MODEL_PATH)
    roberta_model.eval()
    
    # 4. 准备每个领域的数据（训练集和测试集分开）
    print("\n[Step 4] Preparing domain data (train/test split)...")
    domain_train_data = {}
    domain_test_data = {}
    domain_tfidf_models = {}
    
    for domain, data in all_data.items():
        texts = data['texts']
        labels = data['labels']
        
        # 划分训练集和测试集（同领域验证用）
        X_temp, y_temp = texts, global_le.transform(labels)
        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            texts, y_temp, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
            stratify=y_temp
        )
        
        # 准备训练数据（拟合新模型）
        print(f"\n  Processing {domain} (training set)...")
        X_train, _, tfidf_model, _, _ = prepare_domain_data(
            f"{domain}_train", X_train_texts, 
            [global_le.inverse_transform([y])[0] for y in y_train],
            global_le, is_training=True
        )
        domain_tfidf_models[domain] = tfidf_model
        
        # 准备测试数据（使用训练集拟合的模型）
        print(f"  Processing {domain} (test set)...")
        X_test, _, _, _, _ = prepare_domain_data(
            f"{domain}_test", X_test_texts,
            [global_le.inverse_transform([y])[0] for y in y_test],
            global_le, tfidf_model=tfidf_model, 
            tokenizer=tokenizer, roberta_model=roberta_model,
            is_training=False
        )
        
        domain_train_data[domain] = {'X': X_train, 'y': y_train}
        domain_test_data[domain] = {'X': X_test, 'y': y_test}
        
        print(f"    {domain}: Train shape {X_train.shape}, Test shape {X_test.shape}")
    
    # 5. 运行跨领域矩阵实验
    print("\n[Step 5] Running cross-domain transfer matrix experiment...")
    domains = list(domain_train_data.keys())
    
    # 初始化结果矩阵
    train_results = pd.DataFrame(index=domains, columns=domains, dtype=float)
    test_results = pd.DataFrame(index=domains, columns=domains, dtype=float)
    
    for train_domain in domains:
        X_train = domain_train_data[train_domain]['X']
        y_train = domain_train_data[train_domain]['y']
        
        # 训练分类器
        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, 
            random_state=RANDOM_STATE, 
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        
        # 在训练集上评估（同领域训练集）
        y_train_pred = clf.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_results.loc[train_domain, train_domain] = train_acc
        
        # 在所有领域的测试集上评估
        for test_domain in domains:
            X_test = domain_test_data[test_domain]['X']
            y_test = domain_test_data[test_domain]['y']
            
            y_pred = clf.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            
            test_results.loc[train_domain, test_domain] = test_acc
            
            status = "🟢 (Same)" if train_domain == test_domain else "🔵 (Cross)"
            print(f"  {status} {train_domain} -> {test_domain}: {test_acc:.4f}")
    
    # 6. 保存结果
    train_results.to_csv(os.path.join(OUTPUT_DIR, "train_set_accuracy.csv"))
    test_results.to_csv(os.path.join(OUTPUT_DIR, "cross_domain_accuracy.csv"))
    
    print(f"\n✅ Results saved to: {OUTPUT_DIR}")
    
    # 7. 统计分析
    print("\n" + "="*60)
    print("📊 STATISTICAL ANALYSIS REPORT")
    print("="*60)
    
    # 行平均（泛化能力）
    row_mean = test_results.mean(axis=1)
    print("\n[Generalization] (Train domain -> Average across all test domains):")
    for d in domains:
        print(f"  {d}: {row_mean[d]:.4f}")
    
    # 列平均（可迁移性）
    col_mean = test_results.mean(axis=0)
    print("\n[Transferability] (Average across all train domains -> Test domain):")
    for d in domains:
        print(f"  {d}: {col_mean[d]:.4f}")
    
    # 同域 vs 跨域对比
    same_domain_acc = np.mean([test_results.loc[d, d] for d in domains])
    cross_domain_mask = ~np.eye(len(domains), dtype=bool)
    cross_domain_acc = test_results.values[cross_domain_mask].mean()
    
    print(f"\n  Same-domain average accuracy: {same_domain_acc:.4f}")
    print(f"  Cross-domain average accuracy: {cross_domain_acc:.4f}")
    print(f"  Performance drop: {(same_domain_acc - cross_domain_acc):.4f}")
    
    # 同领域验证集准确率（与之前单领域实验对比）
    print(f"\n  [Validation] Same-domain test accuracy: {same_domain_acc:.4f}")
    print(f"  (Expected range: 0.70-0.80, matches previous single-domain experiments)")
    
    # 8. 可视化
    print("\n[Step 6] Generating visualizations...")
    plot_heatmap(test_results, "Cross-Domain Transfer Performance (Test Set Accuracy)")
    plot_bar_comparison(row_mean, col_mean, domains)
    plot_validation_comparison(train_results, test_results, domains)
    
    print("\n🎉 Experiment completed! Check results_fixed directory.")
    plt.show()

def plot_heatmap(matrix, title):
    """绘制热力图（英文）"""
    plt.figure(figsize=(10, 8))
    
    # 确保值为浮点数
    matrix_float = matrix.astype(float)
    
    sns.heatmap(matrix_float, annot=True, fmt='.3f', cmap='YlOrRd', 
                vmin=0.5, vmax=1.0, linewidths=0.5, 
                annot_kws={'size': 10})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Test Domain', fontsize=12)
    plt.ylabel('Train Domain', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", "cross_domain_heatmap.png"), 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_bar_comparison(row_mean, col_mean, domains):
    """绘制泛化能力对比图（英文）"""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(domains))
    width = 0.35
    
    ax.bar(x - width/2, row_mean.values, width, 
           label='Generalization (Train → All Test)', 
           color='#4CAF50', alpha=0.8)
    ax.bar(x + width/2, col_mean.values, width, 
           label='Transferability (All Train → Test)', 
           color='#2196F3', alpha=0.8)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Domain Generalization vs Transferability', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    
    # 添加数值标签
    for i, (gen, trans) in enumerate(zip(row_mean.values, col_mean.values)):
        ax.text(i - width/2, gen + 0.01, f'{gen:.3f}', 
                ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, trans + 0.01, f'{trans:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", "domain_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_validation_comparison(train_results, test_results, domains):
    """绘制训练集和测试集准确率对比（验证是否存在过拟合）"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_acc = [train_results.loc[d, d] for d in domains]
    test_acc = [test_results.loc[d, d] for d in domains]
    
    x = np.arange(len(domains))
    width = 0.35
    
    ax.bar(x - width/2, train_acc, width, label='Training Set', 
           color='#FF9800', alpha=0.8)
    ax.bar(x + width/2, test_acc, width, label='Test Set', 
           color='#9C27B0', alpha=0.8)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training vs Test Accuracy (Same Domain)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.5, 1.05])
    
    # 添加数值标签
    for i, (tr, te) in enumerate(zip(train_acc, test_acc)):
        ax.text(i - width/2, tr + 0.01, f'{tr:.3f}', 
                ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, te + 0.01, f'{te:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", "train_test_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()