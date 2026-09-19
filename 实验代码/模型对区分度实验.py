# -*- coding: utf-8 -*-
"""
实验4：模型对区分度分析（修复版 - 修正模型名称匹配）
功能：探究三个Qwen模型两两之间的可区分性
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import re
from transformers import BertTokenizer, BertModel
import torch
import warnings
warnings.filterwarnings('ignore')

# ================== 🔧 配置 ==================

DATA_FILES = {
    'Sports': r"C:\Users\29276\Desktop\大创实验资料\体育merged_output.json",
    'Technology': r"C:\Users\29276\Desktop\大创实验资料\科技merged_output.json",
    'Education': r"C:\Users\29276\Desktop\大创实验资料\教育merged_output.json",
    'Politics': r"C:\Users\29276\Desktop\大创实验资料\时政merged_output.json"
}

# 模型名称映射（你的JSON中的实际名称）
MODEL_NAMES = {
    'Qwen1.5': 'Qwen1.5-0.5B-Instruct',
    'Qwen2': 'Qwen2-0.5B-Instruct',
    'Qwen2.5': 'Qwen2.5-1.5B-Instruct'
}

ROBERTA_MODEL_PATH = r"C:\Users\29276\Desktop\本地模型库\chinese_roberta_wwm_ext"
OUTPUT_DIR = r"C:\Users\29276\Desktop\大创实验资料\实验4_模型对区分度"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ================== 🛠 数据加载 ==================

def load_json_data(file_path):
    """加载JSON数据 - 适配你的文件结构"""
    texts, labels = [], []
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            content = json.load(f)
            
            # 你的文件是一个列表，每个元素是一个新闻条目
            if isinstance(content, list):
                for item in content:
                    # 每个item有"改写结果"字段，包含三个模型的改写
                    for rewrite in item.get("改写结果", []):
                        texts.append(rewrite["文本"])
                        labels.append(rewrite["模型"])
            
            # 如果是单个对象（兼容性处理）
            elif isinstance(content, dict):
                for rewrite in content.get("改写结果", []):
                    texts.append(rewrite["文本"])
                    labels.append(rewrite["模型"])
        
        print(f"  ✅ 加载成功: {len(texts)} 条")
        
        # 打印模型分布
        model_counts = {}
        for label in labels:
            model_counts[label] = model_counts.get(label, 0) + 1
        print(f"     模型分布: {model_counts}")
        
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
    
    return texts, labels

# ================== 🔬 特征提取 ==================

def extract_surface_features(texts):
    """表层特征"""
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

def extract_roberta_features(texts, tokenizer, model, batch_size=32):
    """RoBERTa特征"""
    features = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_vecs = outputs.last_hidden_state[:, 0, :].numpy()
        features.append(cls_vecs)
    return np.vstack(features)

# ================== 🧪 模型对区分实验 ==================

def get_model_pair_names(pair_key):
    """
    根据简写返回完整的模型名称
    pair_key: '1.5vs2', '1.5vs2.5', '2vs2.5'
    """
    mapping = {
        '1.5vs2': ('Qwen1.5-0.5B-Instruct', 'Qwen2-0.5B-Instruct'),
        '1.5vs2.5': ('Qwen1.5-0.5B-Instruct', 'Qwen2.5-1.5B-Instruct'),
        '2vs2.5': ('Qwen2-0.5B-Instruct', 'Qwen2.5-1.5B-Instruct')
    }
    return mapping.get(pair_key, (None, None))

def prepare_pair_data(texts, labels, model1_full, model2_full, tokenizer, roberta_model):
    """
    准备二分类数据
    model1_full, model2_full: 完整的模型名称
    """
    # 筛选两个模型的数据
    pair_indices = [i for i, label in enumerate(labels) if label in [model1_full, model2_full]]
    
    if len(pair_indices) == 0:
        print(f"  ⚠️ 警告: 没有找到 {model1_full} 或 {model2_full} 的数据")
        return None, None, None, None
    
    pair_texts = [texts[i] for i in pair_indices]
    pair_labels = [labels[i] for i in pair_indices]
    
    # 统计各模型数量
    count1 = sum(1 for l in pair_labels if l == model1_full)
    count2 = sum(1 for l in pair_labels if l == model2_full)
    print(f"  📊 样本分布: {model1_full}: {count1}, {model2_full}: {count2}")
    
    # 重新编码为0/1
    le = LabelEncoder()
    y = le.fit_transform(pair_labels)
    
    # 提取特征
    surface = extract_surface_features(pair_texts)
    
    # TF-IDF (每次实验重新拟合，避免数据泄露)
    tfidf = TfidfVectorizer(tokenizer=lambda x: list(jieba.cut(x)), max_features=100)
    tfidf_feats = tfidf.fit_transform(pair_texts).toarray()
    
    # RoBERTa
    roberta_feats = extract_roberta_features(pair_texts, tokenizer, roberta_model)
    
    # 拼接特征
    X = np.hstack([surface, tfidf_feats, roberta_feats])
    
    return X, y, le, tfidf

def run_pair_experiment(pair_key, all_data, tokenizer, roberta_model):
    """
    运行一对模型的对比实验
    pair_key: '1.5vs2', '1.5vs2.5', '2vs2.5'
    """
    model1_full, model2_full = get_model_pair_names(pair_key)
    
    print(f"\n{'='*60}")
    print(f"🔬 实验: {model1_full} vs {model2_full}")
    print(f"{'='*60}")
    
    domain_results = {}
    
    for domain, data in all_data.items():
        print(f"\n  📍 领域: {domain}")
        
        # 准备该领域的数据
        X, y, le, tfidf = prepare_pair_data(
            data['texts'], data['labels'], 
            model1_full, model2_full, tokenizer, roberta_model
        )
        
        # 如果该领域没有这对模型的数据，跳过
        if X is None:
            print(f"  ⚠️ 跳过 {domain}")
            continue
        
        # 检查样本量是否足够
        if len(y) < 10:
            print(f"  ⚠️ 样本太少 ({len(y)}条)，跳过")
            continue
        
        # 划分训练测试集
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )
        except ValueError as e:
            print(f"  ⚠️ 划分失败: {e}")
            continue
        
        # 训练分类器
        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        # 预测
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        
        # 计算指标
        acc = accuracy_score(y_test, y_pred)
        
        # 对于二分类，AUC需要正类的概率
        if len(np.unique(y)) == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = float('nan')
            
        cm = confusion_matrix(y_test, y_pred)
        
        # 存储结果
        domain_results[domain] = {
            'accuracy': acc,
            'auc': auc,
            'confusion_matrix': cm,
            'n_samples': len(y),
            'n_train': len(y_train),
            'n_test': len(y_test),
            'model1': model1_full,
            'model2': model2_full,
            'class_mapping': {0: le.classes_[0], 1: le.classes_[1]}
        }
        
        print(f"    准确率: {acc:.4f}")
        print(f"    AUC: {auc:.4f}")
        print(f"    训练集: {len(y_train)}, 测试集: {len(y_test)}")
        print(f"    混淆矩阵:")
        print(f"      {le.classes_[0]} → {le.classes_[0]}: {cm[0,0]}, → {le.classes_[1]}: {cm[0,1]}")
        print(f"      {le.classes_[1]} → {le.classes_[0]}: {cm[1,0]}, → {le.classes_[1]}: {cm[1,1]}")
    
    return domain_results

def run_all_pair_experiments():
    """
    运行所有三组模型对实验
    """
    print("="*60)
    print("🚀 实验4：模型对区分度分析")
    print("="*60)
    
    # 1. 加载数据
    print("\n📊 加载数据...")
    all_data = {}
    for domain, path in DATA_FILES.items():
        if not os.path.exists(path):
            print(f"  ⚠️ 文件不存在: {path}")
            continue
        print(f"\n{domain}:")
        texts, labels = load_json_data(path)
        if len(texts) > 0:
            all_data[domain] = {'texts': texts, 'labels': labels}
    
    if len(all_data) == 0:
        print("❌ 错误: 没有成功加载任何数据")
        return None
    
    # 2. 加载RoBERTa
    print("\n🔧 加载RoBERTa模型...")
    try:
        tokenizer = BertTokenizer.from_pretrained(ROBERTA_MODEL_PATH)
        roberta_model = BertModel.from_pretrained(ROBERTA_MODEL_PATH)
        roberta_model.eval()
        print("  ✅ 模型加载成功")
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        return None
    
    # 3. 定义模型对（使用key而不是直接模型名）
    pair_keys = ['1.5vs2', '1.5vs2.5', '2vs2.5']
    pair_display = {
        '1.5vs2': 'Qwen1.5 vs Qwen2',
        '1.5vs2.5': 'Qwen1.5 vs Qwen2.5',
        '2vs2.5': 'Qwen2 vs Qwen2.5'
    }
    
    # 4. 运行实验
    all_results = {}
    for pair_key in pair_keys:
        results = run_pair_experiment(pair_key, all_data, tokenizer, roberta_model)
        all_results[pair_display[pair_key]] = results
        
        # 计算该模型对的平均准确率
        if results:
            accs = [r['accuracy'] for r in results.values()]
            mean_acc = np.mean(accs)
            print(f"\n📊 {pair_display[pair_key]} 平均准确率: {mean_acc:.4f}")
    
    return all_results

# ================== 📊 结果分析 ==================

def analyze_pair_results(all_results):
    """
    分析模型对实验结果
    """
    if all_results is None or len(all_results) == 0:
        print("❌ 没有实验结果可分析")
        return None, None
    
    print("\n" + "="*60)
    print("📊 模型对区分度分析报告")
    print("="*60)
    
    # 1. 构建结果表格
    # 获取所有领域
    all_domains = set()
    for pair_results in all_results.values():
        all_domains.update(pair_results.keys())
    domains = sorted(list(all_domains))
    
    pairs = list(all_results.keys())
    
    # 准确率矩阵
    acc_matrix = pd.DataFrame(index=pairs, columns=domains, dtype=float)
    auc_matrix = pd.DataFrame(index=pairs, columns=domains, dtype=float)
    
    for pair in pairs:
        for domain in domains:
            if domain in all_results[pair]:
                acc_matrix.loc[pair, domain] = all_results[pair][domain]['accuracy']
                auc_matrix.loc[pair, domain] = all_results[pair][domain]['auc']
            else:
                acc_matrix.loc[pair, domain] = float('nan')
                auc_matrix.loc[pair, domain] = float('nan')
    
    print("\n📋 各模型对在不同领域的准确率:")
    print(acc_matrix.round(4))
    
    # 2. 计算平均准确率
    print("\n🔍 平均准确率排名:")
    mean_acc = acc_matrix.mean(axis=1).sort_values(ascending=False)
    for pair, acc in mean_acc.items():
        if not np.isnan(acc):
            print(f"  {pair}: {acc:.4f}")
    
    # 3. 领域特异性分析
    print("\n🌍 领域特异性分析:")
    for domain in domains:
        domain_accs = []
        for pair in pairs:
            if domain in all_results[pair]:
                domain_accs.append(all_results[pair][domain]['accuracy'])
        if domain_accs:
            domain_mean = np.mean(domain_accs)
            domain_std = np.std(domain_accs)
            print(f"  {domain}: 平均 {domain_mean:.4f} ± {domain_std:.4f}")
    
    return acc_matrix, auc_matrix

# ================== 🎨 可视化 ==================

def plot_pair_results(all_results, acc_matrix, auc_matrix):
    """
    可视化模型对实验结果
    """
    if acc_matrix is None or len(acc_matrix) == 0:
        print("❌ 没有数据可可视化")
        return
    
    domains = acc_matrix.columns.tolist()
    pairs = acc_matrix.index.tolist()
    
    # 图1：柱状图 - 各模型对在各领域的准确率
    fig, axes = plt.subplots(1, len(pairs), figsize=(5*len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for idx, (pair, ax) in enumerate(zip(pairs, axes)):
        accs = [acc_matrix.loc[pair, d] for d in domains]
        
        # 过滤掉NaN
        valid_data = [(d, a) for d, a in zip(domains, accs) if not np.isnan(a)]
        if not valid_data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            continue
            
        valid_domains, valid_accs = zip(*valid_data)
        
        bars = ax.bar(valid_domains, valid_accs, color=colors[:len(valid_domains)])
        ax.set_ylim([0.5, 1.0])
        ax.set_title(pair, fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
        
        # 添加数值标签
        for bar, acc in zip(bars, valid_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        if idx == 0:
            ax.legend()
    
    plt.suptitle('Model Pair Discriminability Across Domains', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pair_accuracy_comparison.png'), dpi=300)
    plt.show()
    
    # 图2：热力图 - 准确率矩阵
    plt.figure(figsize=(10, 6))
    
    # 创建用于热力图的数据（去掉NaN行）
    heatmap_data = acc_matrix.dropna(how='all')
    if len(heatmap_data) > 0:
        sns.heatmap(heatmap_data.astype(float), annot=True, fmt='.3f', 
                    cmap='YlOrRd', vmin=0.5, vmax=1.0, 
                    linewidths=0.5, annot_kws={'size': 10})
        plt.title('Model Pair Discriminability Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Domain', fontsize=12)
        plt.ylabel('Model Pair', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'pair_accuracy_matrix.png'), dpi=300)
        plt.show()

# ================== 🚀 主程序 ==================

def main():
    # 1. 运行实验
    all_results = run_all_pair_experiments()
    
    if all_results is None:
        print("❌ 实验失败，请检查数据路径")
        return
    
    # 2. 分析结果
    acc_matrix, auc_matrix = analyze_pair_results(all_results)
    
    if acc_matrix is not None:
        # 3. 保存结果
        acc_matrix.to_csv(os.path.join(OUTPUT_DIR, 'pair_accuracy.csv'))
        auc_matrix.to_csv(os.path.join(OUTPUT_DIR, 'pair_auc.csv'))
        
        # 4. 可视化
        plot_pair_results(all_results, acc_matrix, auc_matrix)
        
        print(f"\n✅ 所有结果已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()