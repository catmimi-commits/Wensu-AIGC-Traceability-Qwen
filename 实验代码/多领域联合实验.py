# -*- coding: utf-8 -*-
"""
实验3：多领域联合训练
功能：验证混合训练能否提升跨领域泛化能力
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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

ROBERTA_MODEL_PATH = r"C:\Users\29276\Desktop\本地模型库\chinese_roberta_wwm_ext"
OUTPUT_DIR = r"C:\Users\29276\Desktop\大创实验资料\联合训练结果"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ================== 🛠 数据加载 ==================

def load_json_data(file_path):
    """加载JSON数据"""
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        content = json.load(f)
        if isinstance(content, dict):
            content = [content]
        for item in content:
            for rewrite in item.get("改写结果", []):
                texts.append(rewrite["文本"])
                labels.append(rewrite["模型"])
    return texts, labels

def load_all_domains():
    """加载所有领域数据"""
    all_data = {}
    print("="*60)
    print("📊 加载所有领域数据")
    print("="*60)
    
    for domain, path in DATA_FILES.items():
        texts, labels = load_json_data(path)
        all_data[domain] = {'texts': texts, 'labels': labels}
        print(f"  ✅ {domain}: {len(texts)} 条样本")
    
    return all_data

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

def prepare_features(texts, tfidf_model=None, tokenizer=None, model=None, fit=False):
    """准备特征（统一接口）"""
    # 表层特征
    surface = extract_surface_features(texts)
    
    # TF-IDF特征
    if fit:
        tfidf = TfidfVectorizer(tokenizer=lambda x: list(jieba.cut(x)), max_features=100)
        tfidf_feats = tfidf.fit_transform(texts).toarray()
    else:
        tfidf_feats = tfidf_model.transform(texts).toarray()
    
    # RoBERTa特征
    roberta_feats = extract_roberta_features(texts, tokenizer, model)
    
    return np.hstack([surface, tfidf_feats, roberta_feats]), tfidf if fit else tfidf_model

# ================== 🚀 联合训练实验 ==================

def create_joint_dataset(domains_to_include, all_data, global_le, tokenizer, model):
    """
    创建联合训练数据集
    domains_to_include: 要包含的领域列表，如 ['Sports', 'Education']
    """
    all_texts = []
    all_labels = []
    domain_origin = []  # 记录每条数据来自哪个领域
    
    print(f"\n🔗 创建联合数据集: {domains_to_include}")
    
    for domain in domains_to_include:
        texts = all_data[domain]['texts']
        labels = all_data[domain]['labels']
        
        # 划分训练集和测试集（每个领域单独划分）
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
        )
        
        # 只取训练集用于联合训练
        all_texts.extend(train_texts)
        all_labels.extend(train_labels)
        domain_origin.extend([domain] * len(train_texts))
        
        print(f"    {domain}: 训练集 {len(train_texts)} 条")
    
    # 编码标签
    y = global_le.transform(all_labels)
    
    # 提取特征（拟合新的TF-IDF）
    X, tfidf_model = prepare_features(all_texts, fit=True, tokenizer=tokenizer, model=model)
    
    return {
        'X': X,
        'y': y,
        'texts': all_texts,
        'labels_raw': all_labels,
        'domain_origin': domain_origin,
        'tfidf_model': tfidf_model
    }

def evaluate_on_all_domains(clf, all_data, global_le, tokenizer, model, tfidf_model):
    """在所有领域的测试集上评估"""
    results = {}
    
    for domain, data in all_data.items():
        # 划分测试集（用同样的随机种子）
        _, test_texts, _, test_labels = train_test_split(
            data['texts'], data['labels'], test_size=TEST_SIZE, 
            random_state=RANDOM_STATE, stratify=data['labels']
        )
        
        # 提取特征（使用训练时的TF-IDF）
        X_test, _ = prepare_features(test_texts, tfidf_model=tfidf_model, 
                                     tokenizer=tokenizer, model=model, fit=False)
        y_test = global_le.transform(test_labels)
        
        # 预测
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # 详细报告
        report = classification_report(y_test, y_pred, target_names=global_le.classes_, output_dict=True)
        
        results[domain] = {
            'accuracy': acc,
            'report': report,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        print(f"    {domain}: {acc:.4f}")
    
    return results

def run_joint_training_experiment():
    """运行联合训练实验主流程"""
    
    print("\n" + "="*60)
    print("🚀 实验3：多领域联合训练")
    print("="*60)
    
    # 1. 加载数据
    all_data = load_all_domains()
    
    # 2. 加载RoBERTa
    print("\n🔧 加载RoBERTa模型...")
    tokenizer = BertTokenizer.from_pretrained(ROBERTA_MODEL_PATH)
    model = BertModel.from_pretrained(ROBERTA_MODEL_PATH)
    model.eval()
    
    # 3. 构建全局标签编码器
    all_labels = []
    for data in all_data.values():
        all_labels.extend(data['labels'])
    global_le = LabelEncoder()
    global_le.fit(all_labels)
    print(f"\n🏷️ 模型类别: {list(global_le.classes_)}")
    
    # 4. 定义实验配置（新增F组：仅使用时政领域）
    experiments = {
        'A_Sports_Only': ['Sports'],
        'B_Tech_Only': ['Technology'],
        'C_Edu_Politics': ['Education', 'Politics'],
        'D_Without_Tech': ['Sports', 'Education', 'Politics'],
        'E_All_Domains': ['Sports', 'Technology', 'Education', 'Politics'],
        'F_Politics_Only': ['Politics']  # 新增：仅使用时政领域
    }
    
    # 5. 存储所有结果
    all_results = {}
    
    # 6. 运行每个实验
    for exp_name, domains in experiments.items():
        print("\n" + "-"*60)
        print(f"📌 实验: {exp_name}")
        print(f"   训练领域: {domains}")
        print("-"*60)
        
        # 创建联合训练集
        joint_data = create_joint_dataset(domains, all_data, global_le, tokenizer, model)
        
        # 训练分类器
        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(joint_data['X'], joint_data['y'])
        
        # 训练集准确率
        train_acc = accuracy_score(joint_data['y'], clf.predict(joint_data['X']))
        print(f"\n📈 训练集准确率: {train_acc:.4f}")
        
        # 在所有领域测试
        print("\n🌍 跨领域测试结果:")
        test_results = evaluate_on_all_domains(
            clf, all_data, global_le, tokenizer, model, joint_data['tfidf_model']
        )
        
        all_results[exp_name] = {
            'train_domains': domains,
            'train_acc': train_acc,
            'test_results': test_results,
            'clf': clf,
            'joint_data': joint_data
        }
    
    return all_results, global_le

# ================== 📊 结果分析 ==================

def analyze_results(all_results, global_le):
    """分析实验结果"""
    
    print("\n" + "="*60)
    print("📊 联合训练结果分析")
    print("="*60)
    
    # 1. 构建结果矩阵
    domains = list(DATA_FILES.keys())
    experiments = list(all_results.keys())
    
    # 准确率矩阵
    acc_matrix = pd.DataFrame(index=experiments, columns=domains)
    
    for exp_name, exp_data in all_results.items():
        for domain in domains:
            acc_matrix.loc[exp_name, domain] = exp_data['test_results'][domain]['accuracy']
    
    print("\n📋 各实验跨领域准确率矩阵:")
    print(acc_matrix.round(4))
    
    # 2. 计算关键指标
    print("\n🔍 关键指标分析:")
    
    for exp_name in experiments:
        exp_data = all_results[exp_name]
        train_domains = exp_data['train_domains']
        test_results = exp_data['test_results']
        
        # 同领域平均（训练集包含的领域）
        same_domain_acc = np.mean([test_results[d]['accuracy'] for d in train_domains])
        
        # 跨领域平均（训练集不包含的领域）
        cross_domains = [d for d in domains if d not in train_domains]
        if cross_domains:
            cross_domain_acc = np.mean([test_results[d]['accuracy'] for d in cross_domains])
        else:
            cross_domain_acc = np.nan
        
        print(f"\n  {exp_name}:")
        print(f"    训练领域: {train_domains}")
        print(f"    同领域平均: {same_domain_acc:.4f}")
        if not np.isnan(cross_domain_acc):
            print(f"    跨领域平均: {cross_domain_acc:.4f}")
            print(f"    泛化差距: {same_domain_acc - cross_domain_acc:.4f}")
    
    # 3. 计算"科技孤岛"效应
    print("\n🔬 科技孤岛效应分析:")
    for exp_name in experiments:
        if 'Technology' not in all_results[exp_name]['train_domains']:
            tech_acc = all_results[exp_name]['test_results']['Technology']['accuracy']
            print(f"  {exp_name} → 科技: {tech_acc:.4f}")
    
    # 4. 计算"时政孤岛"效应（新增）
    print("\n🔬 时政孤岛效应分析:")
    for exp_name in experiments:
        if 'Politics' not in all_results[exp_name]['train_domains']:
            politics_acc = all_results[exp_name]['test_results']['Politics']['accuracy']
            print(f"  {exp_name} → 时政: {politics_acc:.4f}")
    
    return acc_matrix

# ================== 🎨 可视化 ==================

def plot_joint_training_results(all_results, acc_matrix):
    """可视化联合训练结果"""
    
    # 图1：热力图 - 各实验跨领域表现
    plt.figure(figsize=(12, 8))
    sns.heatmap(acc_matrix.astype(float), annot=True, fmt='.3f', 
                cmap='YlOrRd', vmin=0.3, vmax=0.9, 
                linewidths=0.5, annot_kws={'size': 10})
    plt.title('Multi-Domain Joint Training Results', fontsize=14, fontweight='bold')
    plt.xlabel('Test Domain', fontsize=12)
    plt.ylabel('Training Experiment', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'joint_training_heatmap.png'), dpi=300)
    plt.show()
    
    # 图2：柱状对比 - 同领域 vs 跨领域
    fig, ax = plt.subplots(figsize=(16, 6))
    
    experiments = list(acc_matrix.index)
    x = np.arange(len(experiments))
    width = 0.35
    
    # 计算同领域和跨领域平均
    same_domain_means = []
    cross_domain_means = []
    
    for exp in experiments:
        train_domains = all_results[exp]['train_domains']
        domains = acc_matrix.columns
        
        same = [acc_matrix.loc[exp, d] for d in train_domains]
        cross = [acc_matrix.loc[exp, d] for d in domains if d not in train_domains]
        
        same_domain_means.append(np.mean(same))
        cross_domain_means.append(np.mean(cross) if cross else 0)
    
    ax.bar(x - width/2, same_domain_means, width, label='Same-Domain Avg', color='#4CAF50', alpha=0.8)
    ax.bar(x + width/2, cross_domain_means, width, label='Cross-Domain Avg', color='#FF9800', alpha=0.8)
    
    ax.set_xlabel('Training Experiment', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Same-Domain vs Cross-Domain Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # 添加数值标签
    for i, (same, cross) in enumerate(zip(same_domain_means, cross_domain_means)):
        ax.text(i - width/2, same + 0.02, f'{same:.3f}', ha='center', va='bottom', fontsize=9)
        if cross > 0:
            ax.text(i + width/2, cross + 0.02, f'{cross:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'joint_training_comparison.png'), dpi=300)
    plt.show()
    
    # 图3：雷达图 - 各实验的领域覆盖能力
    from math import pi
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    domains = list(acc_matrix.columns)
    N = len(domains)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for idx, exp in enumerate(experiments):
        values = acc_matrix.loc[exp].values.tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=exp, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(domains, fontsize=11)
    ax.set_ylim([0, 1])
    ax.set_title('Domain Coverage Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'joint_training_radar.png'), dpi=300)
    plt.show()

def plot_island_analysis(all_results):
    """科技孤岛和时政孤岛效应专门分析（更新）"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    experiments = list(all_results.keys())
    
    for idx, exp in enumerate(experiments):
        if idx >= 6:
            break
            
        ax = axes[idx]
        test_results = all_results[exp]['test_results']
        
        domains = list(test_results.keys())
        accs = [test_results[d]['accuracy'] for d in domains]
        
        # 颜色设置：科技用红色，时政用紫色，其他用青色
        colors = []
        for d in domains:
            if d == 'Technology':
                colors.append('#FF6B6B')  # 红色
            elif d == 'Politics':
                colors.append('#9B59B6')  # 紫色
            else:
                colors.append('#4ECDC4')  # 青色
        
        bars = ax.bar(domains, accs, color=colors)
        ax.set_title(f'{exp}', fontsize=12)
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0, 1])
        ax.axhline(y=0.333, color='gray', linestyle='--', alpha=0.5, label='Random (33.3%)')
        
        # 添加数值标签
        for i, (bar, acc) in enumerate(zip(bars, accs)):
            ax.text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=9, fontweight='bold')
        
        if idx == 0:
            ax.legend()
    
    plt.suptitle('Domain Island Effect Analysis (Technology in Red, Politics in Purple)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'island_analysis.png'), dpi=300)
    plt.show()

# ================== 🚀 主程序 ==================

def main():
    # 1. 运行联合训练实验
    all_results, global_le = run_joint_training_experiment()
    
    # 2. 分析结果
    acc_matrix = analyze_results(all_results, global_le)
    
    # 3. 保存结果
    acc_matrix.to_csv(os.path.join(OUTPUT_DIR, 'joint_training_results.csv'))
    
    # 4. 可视化
    print("\n🎨 生成可视化图表...")
    plot_joint_training_results(all_results, acc_matrix)
    plot_island_analysis(all_results)
    
    print(f"\n✅ 所有结果已保存至: {OUTPUT_DIR}")
    print("\n" + "="*60)
    print("📝 论文写作建议")
    print("="*60)
   

if __name__ == "__main__":
    main()