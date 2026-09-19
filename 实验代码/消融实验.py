# -*- coding: utf-8 -*-
"""
实验五：特征消融实验
功能：探究不同特征对模型指纹识别的贡献度
- 表层特征 (6维)
- TF-IDF特征 (100维)  
- RoBERTa特征 (768维)
"""

import json
import jieba
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import torch
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ================== 配置 ==================

# 四个领域的数据文件
DATA_FILES = {
    'Sports': r"C:\Users\29276\Desktop\大创实验资料\体育merged_output.json",
    'Technology': r"C:\Users\29276\Desktop\大创实验资料\科技merged_output.json",
    'Education': r"C:\Users\29276\Desktop\大创实验资料\教育merged_output.json",
    'Politics': r"C:\Users\29276\Desktop\大创实验资料\时政merged_output.json"
}

ROBERTA_MODEL_NAME = r"C:\Users\29276\Desktop\本地模型库\chinese_roberta_wwm_ext"
OUTPUT_DIR = r"C:\Users\29276\Desktop\大创实验资料\实验五_消融实验"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ================== 数据加载 ==================

def load_domain_data(file_path, domain_name):
    """加载单个领域的数据"""
    texts, models = [], []
    
    print(f"加载 {domain_name} 数据...")
    
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        content = json.load(f)
        
        if isinstance(content, list):
            for item in content:
                for rewrite in item.get("改写结果", []):
                    texts.append(rewrite["文本"])
                    models.append(rewrite["模型"])
        elif isinstance(content, dict):
            for rewrite in content.get("改写结果", []):
                texts.append(rewrite["文本"])
                models.append(rewrite["模型"])
    
    print(f"  → 加载 {len(texts)} 条")
    return texts, models

def load_all_data():
    """加载所有领域数据"""
    all_data = {}
    all_texts, all_models, all_domains = [], [], []
    
    print("="*60)
    print("📊 加载所有领域数据")
    print("="*60)
    
    for domain, path in DATA_FILES.items():
        texts, models = load_domain_data(path, domain)
        all_data[domain] = {'texts': texts, 'models': models}
        all_texts.extend(texts)
        all_models.extend(models)
        all_domains.extend([domain] * len(texts))
    
    print(f"\n✅ 总计: {len(all_texts)} 条文本")
    return all_data, all_texts, all_models, all_domains

# ================== 特征提取 ==================

def extract_surface_features_batch(texts):
    """批量提取表层特征 (6维)"""
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

def extract_tfidf_features_batch(texts, max_features=100):
    """批量提取TF-IDF特征 (100维)"""
    tfidf = TfidfVectorizer(
        tokenizer=lambda x: list(jieba.cut(x)),
        max_features=max_features,
        stop_words=['的', '了', '在', '是', '和', '与']
    )
    features = tfidf.fit_transform(texts).toarray()
    return features, tfidf

def extract_roberta_features_batch(texts, tokenizer, model, batch_size=32):
    """批量提取RoBERTa特征 (768维)"""
    features = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_vecs = outputs.last_hidden_state[:, 0, :].numpy()
        features.append(cls_vecs)
    return np.vstack(features)

# ================== 特征组合 ==================

def get_feature_combinations(texts, tokenizer, roberta_model):
    """
    提取所有特征组合
    返回字典: 组合名称 -> 特征矩阵
    """
    print("\n🔧 提取各类特征...")
    
    # 1. 提取各类特征
    print("  提取表层特征...")
    surface = extract_surface_features_batch(texts)
    
    print("  提取TF-IDF特征...")
    tfidf, tfidf_model = extract_tfidf_features_batch(texts)
    
    print("  提取RoBERTa特征...")
    roberta = extract_roberta_features_batch(texts, tokenizer, roberta_model)
    
    # 2. 定义所有组合
    combinations = {
        'Full': np.hstack([surface, tfidf, roberta]),           # 6+100+768=874
        'No_Surface': np.hstack([tfidf, roberta]),              # 100+768=868
        'No_TFIDF': np.hstack([surface, roberta]),              # 6+768=774
        'No_RoBERTa': np.hstack([surface, tfidf]),              # 6+100=106
        'Only_Surface': surface,                                 # 6
        'Only_TFIDF': tfidf,                                     # 100
        'Only_RoBERTa': roberta,                                 # 768
        'Surface_TFIDF': np.hstack([surface, tfidf]),           # 106 (同No_RoBERTa)
        'Surface_RoBERTa': np.hstack([surface, roberta]),       # 774 (同No_TFIDF)
        'TFIDF_RoBERTa': np.hstack([tfidf, roberta])            # 868 (同No_Surface)
    }
    
    # 打印各组合维度
    print("\n📐 特征组合维度:")
    for name, feat in combinations.items():
        print(f"  {name:15}: {feat.shape[1]} 维")
    
    return combinations, tfidf_model

# ================== 消融实验 ==================

def run_ablation_experiment(domain_data, domain_name, tokenizer, roberta_model):
    """
    在单个领域上运行所有消融实验
    """
    print(f"\n{'='*60}")
    print(f"🔬 消融实验 - {domain_name}")
    print(f"{'='*60}")
    
    texts = domain_data['texts']
    models = domain_data['models']
    
    # 编码标签
    le = LabelEncoder()
    y = le.fit_transform(models)
    
    # 划分训练测试集
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # 提取特征（必须在训练集上拟合！）
    print("\n📊 在训练集上提取特征...")
    
    # 训练集特征
    train_combinations, tfidf_model = get_feature_combinations(
        X_train_texts, tokenizer, roberta_model
    )
    
    # 测试集特征（使用训练集拟合的TF-IDF）
    print("\n📊 在测试集上提取特征...")
    test_combinations = {}
    
    # 测试集的表层特征
    test_surface = extract_surface_features_batch(X_test_texts)
    
    # 测试集的TF-IDF（用训练集拟合的模型）
    test_tfidf = tfidf_model.transform(X_test_texts).toarray()
    
    # 测试集的RoBERTa
    test_roberta = extract_roberta_features_batch(X_test_texts, tokenizer, roberta_model)
    
    # 构建测试集的所有组合
    test_combinations['Full'] = np.hstack([test_surface, test_tfidf, test_roberta])
    test_combinations['No_Surface'] = np.hstack([test_tfidf, test_roberta])
    test_combinations['No_TFIDF'] = np.hstack([test_surface, test_roberta])
    test_combinations['No_RoBERTa'] = np.hstack([test_surface, test_tfidf])
    test_combinations['Only_Surface'] = test_surface
    test_combinations['Only_TFIDF'] = test_tfidf
    test_combinations['Only_RoBERTa'] = test_roberta
    test_combinations['Surface_TFIDF'] = np.hstack([test_surface, test_tfidf])
    test_combinations['Surface_RoBERTa'] = np.hstack([test_surface, test_roberta])
    test_combinations['TFIDF_RoBERTa'] = np.hstack([test_tfidf, test_roberta])
    
    # 运行实验
    results = {}
    
    print("\n🏃 运行分类实验...")
    for feat_name, X_train in train_combinations.items():
        X_test = test_combinations[feat_name]
        
        # 训练分类器
        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        # 预测
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        results[feat_name] = {
            'accuracy': acc,
            'dimension': X_train.shape[1],
            'feature_name': feat_name
        }
        
        print(f"  {feat_name:15}: {acc:.4f} ({X_train.shape[1]:3d}维)")
    
    return results, le.classes_

# ================== 结果分析 ==================

def analyze_ablation_results(all_results):
    """
    分析所有领域的消融实验结果
    """
    print("\n" + "="*60)
    print("📊 消融实验综合分析报告")
    print("="*60)
    
    # 构建结果表格
    domains = list(all_results.keys())
    features = ['Full', 'No_Surface', 'No_TFIDF', 'No_RoBERTa', 
                'Only_Surface', 'Only_TFIDF', 'Only_RoBERTa']
    
    # 特征组合的显示名称
    feature_names = {
        'Full': 'Full (All Features)',
        'No_Surface': 'w/o Surface',
        'No_TFIDF': 'w/o TF-IDF',
        'No_RoBERTa': 'w/o RoBERTa',
        'Only_Surface': 'Only Surface',
        'Only_TFIDF': 'Only TF-IDF',
        'Only_RoBERTa': 'Only RoBERTa'
    }
    
    # 创建结果矩阵
    results_matrix = pd.DataFrame(index=feature_names.values(), columns=domains)
    
    for domain in domains:
        for feat_key, feat_name in feature_names.items():
            if feat_key in all_results[domain]:
                results_matrix.loc[feat_name, domain] = all_results[domain][feat_key]['accuracy']
    
    print("\n📋 各领域消融实验结果:")
    print(results_matrix.round(4))
    
    # 计算平均准确率
    print("\n🔍 平均准确率排名:")
    mean_acc = results_matrix.mean(axis=1).sort_values(ascending=False)
    for feat, acc in mean_acc.items():
        print(f"  {feat:20}: {acc:.4f}")
    
    # 计算下降幅度（相对于Full）
    print("\n📉 特征重要性分析（移除后下降幅度）:")
    full_acc = mean_acc['Full (All Features)']
    
    下降幅度 = {
        'Remove Surface': full_acc - mean_acc['w/o Surface'],
        'Remove TF-IDF': full_acc - mean_acc['w/o TF-IDF'],
        'Remove RoBERTa': full_acc - mean_acc['w/o RoBERTa']
    }
    
    for feat, drop in 下降幅度.items():
        print(f"  {feat:15}: {drop:.4f}")
    
    # 找出最重要的特征
    most_important = max(下降幅度, key=下降幅度.get)
    print(f"\n🎯 最重要的特征: {most_important} (下降{下降幅度[most_important]:.4f})")
    
    return results_matrix, 下降幅度

# ================== 可视化 ==================

def setup_english_plot():
    """设置英文绘图环境"""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1.2

def plot_ablation_results(results_matrix, drop_dict, save_path):
    """
    绘制消融实验结果
    """
    setup_english_plot()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ===== 图1：各领域柱状对比 =====
    ax1 = axes[0, 0]
    
    domains = results_matrix.columns
    features = ['Full', 'w/o Surface', 'w/o TF-IDF', 'w/o RoBERTa', 
                'Only Surface', 'Only TF-IDF', 'Only RoBERTa']
    
    x = np.arange(len(domains))
    width = 0.12
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51', '#5E4B56']
    
    for i, feat in enumerate(features):
        accs = [results_matrix.loc[feat, d] for d in domains]
        ax1.bar(x + i*width, accs, width, label=feat, color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Domain', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Ablation Results Across Domains', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * 3)
    ax1.set_xticklabels(domains)
    ax1.legend(loc='lower left', fontsize=9, ncol=2)
    ax1.set_ylim([0.5, 0.9])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ===== 图2：特征重要性 =====
    ax2 = axes[0, 1]
    
    features_imp = list(drop_dict.keys())
    drops = list(drop_dict.values())
    
    bars = ax2.barh(features_imp, drops, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_xlabel('Accuracy Drop', fontsize=12)
    ax2.set_title('Feature Importance (Drop when Removed)', fontsize=14, fontweight='bold')
    
    for bar, drop in zip(bars, drops):
        ax2.text(drop + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{drop:.4f}', va='center', fontsize=10)
    
    ax2.grid(True, alpha=0.3, axis='x')
    
    # ===== 图3：各领域最佳特征 =====
    ax3 = axes[1, 0]
    
    # 计算各领域的最佳单特征
    best_single = {}
    for domain in domains:
        only_accs = {
            'Surface': results_matrix.loc['Only Surface', domain],
            'TF-IDF': results_matrix.loc['Only TF-IDF', domain],
            'RoBERTa': results_matrix.loc['Only RoBERTa', domain]
        }
        best_single[domain] = max(only_accs, key=only_accs.get)
    
    # 绘制饼图
    from collections import Counter
    best_counts = Counter(best_single.values())
    
    wedges, texts, autotexts = ax3.pie(best_counts.values(), 
                                        labels=best_counts.keys(),
                                        autopct='%1.1f%%',
                                        colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax3.set_title('Best Single Feature by Domain', fontsize=14, fontweight='bold')
    
    # ===== 图4：雷达图 =====
    ax4 = axes[1, 1]
    
    # 准备雷达图数据
    from math import pi
    angles = np.linspace(0, 2 * np.pi, len(domains), endpoint=False).tolist()
    angles += angles[:1]
    
    features_radar = ['Full', 'Only Surface', 'Only TF-IDF', 'Only RoBERTa']
    
    for feat in features_radar:
        values = results_matrix.loc[feat].values.tolist()
        values += values[:1]
        ax4.plot(angles, values, 'o-', linewidth=2, label=feat)
        ax4.fill(angles, values, alpha=0.1)
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(domains)
    ax4.set_ylim([0.5, 0.9])
    ax4.set_title('Feature Performance Radar', fontsize=14, fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_feature_heatmap(results_matrix, save_path):
    """
    绘制特征热力图
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(results_matrix.astype(float), annot=True, fmt='.3f',
                cmap='YlOrRd', vmin=0.5, vmax=0.9,
                linewidths=0.5, annot_kws={'size': 10})
    
    plt.title('Ablation Study Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Domain', fontsize=12)
    plt.ylabel('Feature Combination', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# ================== 主程序 ==================

def main():
    print("="*60)
    print("🚀 实验五：特征消融实验")
    print("="*60)
    
    # 1. 加载数据
    all_data, _, _, _ = load_all_data()
    
    # 2. 加载RoBERTa
    print("\n🔧 加载RoBERTa模型...")
    tokenizer = BertTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    roberta_model = BertModel.from_pretrained(ROBERTA_MODEL_NAME)
    roberta_model.eval()
    
    # 3. 运行各领域的消融实验
    all_results = {}
    all_classes = {}
    
    for domain, data in all_data.items():
        results, classes = run_ablation_experiment(data, domain, tokenizer, roberta_model)
        all_results[domain] = results
        all_classes[domain] = classes
    
    # 4. 分析结果
    results_matrix, drop_dict = analyze_ablation_results(all_results)
    
    # 5. 保存结果
    results_matrix.to_csv(os.path.join(OUTPUT_DIR, 'ablation_results.csv'))
    
    # 6. 可视化
    print("\n🎨 生成可视化图表...")
    
    main_plot_path = os.path.join(OUTPUT_DIR, 'ablation_main.png')
    plot_ablation_results(results_matrix, drop_dict, main_plot_path)
    
    heatmap_path = os.path.join(OUTPUT_DIR, 'ablation_heatmap.png')
    plot_feature_heatmap(results_matrix, heatmap_path)
    
    # 7. 生成论文表格
    print("\n📝 论文表格（LaTeX格式）:")
    print("="*60)
    
    print("""
\\begin{table}[t]
\\centering
\\caption{Ablation Study Results}
\\label{tab:ablation}
\\begin{tabular}{lcccc}
\\hline
\\textbf{Features} & \\textbf{Sports} & \\textbf{Tech} & \\textbf{Education} & \\textbf{Politics} \\\\
\\hline""")
    
    for feat in ['Full', 'No_Surface', 'No_TFIDF', 'No_RoBERTa', 
                 'Only_Surface', 'Only_TFIDF', 'Only_RoBERTa']:
        row = f"\\textbf{{{feat.replace('_', ' ')}}}"
        for domain in results_matrix.columns:
            if feat in all_results[domain]:
                acc = all_results[domain][feat]['accuracy']
                row += f" & {acc:.3f}"
            else:
                row += " & -"
        row += " \\\\"
        print(row)
    
    print("""\\hline
\\end{tabular}
\\end{table}""")
    
    print(f"\n✅ 所有结果已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()