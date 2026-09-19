# -*- coding: utf-8 -*-
"""
t-SNE可视化：4领域 × 3模型
功能：同时显示领域（颜色）和模型（形状）
输出：英文标签的出版级图片
"""

import json
import jieba
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import random
from collections import defaultdict
import os

# ================== 配置 ==================
# 四个领域的数据文件
DATA_FILES = {
    'Sports': r"C:\Users\29276\Desktop\大创实验资料\体育merged_output.json",
    'Technology': r"C:\Users\29276\Desktop\大创实验资料\科技merged_output.json",
    'Education': r"C:\Users\29276\Desktop\大创实验资料\教育merged_output.json",
    'Politics': r"C:\Users\29276\Desktop\大创实验资料\时政merged_output.json"
}

ROBERTA_MODEL_NAME = r"C:\Users\29276\Desktop\本地模型库\chinese_roberta_wwm_ext"
OUTPUT_DIR = r"C:\Users\29276\Desktop\大创实验资料\tSNE_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 采样参数（每个领域每个模型采多少条）
SAMPLES_PER_GROUP = 50  # 3模型 × 4领域 × 50 = 600条，既清晰又不拥挤
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ================== 数据加载 ==================

def load_domain_data(file_path, domain_name):
    """加载单个领域的数据"""
    texts, models = [], []
    
    print(f"加载 {domain_name} 数据...")
    
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        content = json.load(f)
        
        # 处理列表格式
        if isinstance(content, list):
            for item in content:
                for rewrite in item.get("改写结果", []):
                    texts.append(rewrite["文本"])
                    models.append(rewrite["模型"])
        
        # 处理单个对象格式
        elif isinstance(content, dict):
            for rewrite in content.get("改写结果", []):
                texts.append(rewrite["文本"])
                models.append(rewrite["模型"])
    
    # 统计
    model_counts = {}
    for m in models:
        model_counts[m] = model_counts.get(m, 0) + 1
    
    print(f"  → 总计 {len(texts)} 条")
    print(f"  → 分布: {model_counts}")
    
    return texts, models, [domain_name] * len(texts)

def load_all_data():
    """加载所有领域数据"""
    all_texts = []
    all_models = []
    all_domains = []
    
    print("="*60)
    print("📊 加载所有领域数据")
    print("="*60)
    
    for domain, path in DATA_FILES.items():
        texts, models, domains = load_domain_data(path, domain)
        all_texts.extend(texts)
        all_models.extend(models)
        all_domains.extend(domains)
    
    print(f"\n✅ 总计加载: {len(all_texts)} 条文本")
    print(f"   涉及模型: {set(all_models)}")
    print(f"   涉及领域: {set(all_domains)}")
    
    return all_texts, all_models, all_domains

# ================== 分层采样 ==================

def stratified_sample(texts, models, domains, samples_per_group=50):
    """
    分层采样：每个(模型, 领域)组合抽取固定数量样本
    """
    # 按(模型, 领域)分组
    groups = defaultdict(list)
    for i, (text, model, domain) in enumerate(zip(texts, models, domains)):
        groups[(model, domain)].append(i)
    
    sampled_indices = []
    sampling_stats = {}
    
    print("\n📊 采样统计:")
    print("-" * 40)
    
    for (model, domain), indices in sorted(groups.items()):
        n_available = len(indices)
        n_sample = min(samples_per_group, n_available)
        
        # 随机采样
        sampled = random.sample(indices, n_sample)
        sampled_indices.extend(sampled)
        
        sampling_stats[f"{model}|{domain}"] = {
            'available': n_available,
            'sampled': n_sample
        }
        
        print(f"{model[:10]}... | {domain:10} : {n_sample}/{n_available}")
    
    print("-" * 40)
    print(f"总计采样: {len(sampled_indices)} 条")
    
    # 根据采样索引提取数据
    sampled_texts = [texts[i] for i in sampled_indices]
    sampled_models = [models[i] for i in sampled_indices]
    sampled_domains = [domains[i] for i in sampled_indices]
    
    return sampled_texts, sampled_models, sampled_domains, sampling_stats

# ================== 特征提取 ==================

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
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_vecs = outputs.last_hidden_state[:, 0, :].numpy()
        features.append(cls_vecs)
    return np.vstack(features)

def extract_all_features(texts, tokenizer, model):
    """提取所有特征并拼接"""
    print("\n🔧 提取特征...")
    
    # 1. 表层特征
    print("  提取表层特征...")
    surface_feats = extract_surface_features_batch(texts)
    
    # 2. TF-IDF特征
    print("  提取TF-IDF特征...")
    tfidf = TfidfVectorizer(tokenizer=lambda x: list(jieba.cut(x)), max_features=100)
    tfidf_feats = tfidf.fit_transform(texts).toarray()
    
    # 3. RoBERTa特征
    print("  提取RoBERTa特征 (这可能需要几分钟)...")
    roberta_feats = extract_roberta_features_batch(texts, tokenizer, model)
    
    # 拼接
    X = np.hstack([surface_feats, tfidf_feats, roberta_feats])
    print(f"✅ 特征矩阵形状: {X.shape}")
    
    return X, tfidf

# ================== t-SNE可视化 ==================

def setup_english_plot():
    """设置英文绘图环境"""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2

def plot_tsne_4domains_3models(X, models, domains, save_path):
    """
    绘制t-SNE图：4种颜色代表领域，3种形状代表模型
    """
    print("\n🎨 生成t-SNE降维...")
    
    # t-SNE降维
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # ===== 左图：按领域着色 =====
    print("  绘制左图（按领域着色）...")
    
    # 领域颜色映射
    domain_colors = {
    'Sports': '#D55E00',       # 橙红色
    'Technology': '#0072B2',   # 深蓝色
    'Education': '#F0E442',    # 明黄色
    'Politics': '#CC79A7'      # 粉紫色
}
    
    # 模型形状映射
    model_markers = {
        'Qwen1.5-0.5B-Instruct': 'o',     # 圆圈
        'Qwen2-0.5B-Instruct': 's',       # 方块
        'Qwen2.5-1.5B-Instruct': '^'      # 三角形
    }
    
    # 绘制所有点（左图）
    for model in model_markers.keys():
        for domain in domain_colors.keys():
            mask = (np.array(models) == model) & (np.array(domains) == domain)
            if np.any(mask):
                ax1.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                          c=domain_colors[domain],
                          marker=model_markers[model],
                          label=f'{domain} - {model}',
                          alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
    
    ax1.set_title('t-SNE Visualization: 4 Domains × 3 Models', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('t-SNE Component 1', fontsize=12)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 创建图例（左图）- 按领域
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = []
    # 领域图例（颜色）
    for domain, color in domain_colors.items():
        legend_elements.append(Patch(facecolor=color, alpha=0.7, label=domain))
    # 模型图例（形状）
    for model, marker in model_markers.items():
        model_short = model.replace('-0.5B-Instruct', '').replace('-1.5B-Instruct', '')
        legend_elements.append(Line2D([0], [0], marker=marker, color='gray', 
                                      linestyle='None', markersize=8, label=model_short))
    
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    # ===== 右图：简化版 - 只按模型着色 =====
    print("  绘制右图（按模型着色）...")
    
    # 模型颜色映射（用于右图）
    model_colors = {
        'Qwen1.5-0.5B-Instruct': '#FF6B6B',   # 红色
        'Qwen2-0.5B-Instruct': '#F0E442',      # 黄色
        'Qwen2.5-1.5B-Instruct': '#45B7D1'     # 蓝色
    }
    
    for model in model_colors.keys():
        mask = np.array(models) == model
        if np.any(mask):
            ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                       c=model_colors[model],
                       label=model.replace('-0.5B-Instruct', '').replace('-1.5B-Instruct', ''),
                       alpha=0.6, s=30, edgecolors='white', linewidth=0.3)
    
    ax2.set_title('t-SNE Visualization (Colored by Model)', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('t-SNE Component 1', fontsize=12)
    ax2.set_ylabel('t-SNE Component 2', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存高分辨率图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 图片已保存: {save_path}")
    
    plt.show()
    plt.close()
    
    return X_tsne

def plot_density_comparison(X_tsne, models, domains, save_path):
    """
    绘制密度图，展示各领域的分布密度
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    domains_list = ['Sports', 'Technology', 'Education', 'Politics']
    domain_titles = ['Sports Domain', 'Technology Domain', 'Education Domain', 'Politics Domain']
    
    for idx, (domain, title) in enumerate(zip(domains_list, domain_titles)):
        ax = axes[idx]
        
        # 筛选该领域的数据
        domain_mask = np.array(domains) == domain
        
        # 绘制该领域所有模型的密度图
        for model in ['Qwen1.5-0.5B-Instruct', 'Qwen2-0.5B-Instruct', 'Qwen2.5-1.5B-Instruct']:
            model_mask = np.array(models) == model
            mask = domain_mask & model_mask
            
            if np.any(mask):
                model_short = model.replace('-0.5B-Instruct', '').replace('-1.5B-Instruct', '')
                ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                          alpha=0.6, s=25, label=model_short)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Domain-Specific t-SNE Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# ================== 主程序 ==================

def main():
    print("="*60)
    print("🚀 t-SNE可视化：4领域 × 3模型")
    print("="*60)
    
    # 1. 设置英文环境
    setup_english_plot()
    
    # 2. 加载所有数据
    all_texts, all_models, all_domains = load_all_data()
    
    # 3. 分层采样（每个模型×领域采50条）
    sampled_texts, sampled_models, sampled_domains, stats = stratified_sample(
        all_texts, all_models, all_domains, SAMPLES_PER_GROUP
    )
    
    # 4. 加载RoBERTa模型
    print("\n🔧 加载RoBERTa模型...")
    tokenizer = BertTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    roberta_model = BertModel.from_pretrained(ROBERTA_MODEL_NAME)
    roberta_model.eval()
    
    # 5. 提取特征
    X, tfidf = extract_all_features(sampled_texts, tokenizer, roberta_model)
    
    # 6. 绘制主t-SNE图
    tsne_main_path = os.path.join(OUTPUT_DIR, 'tsne_4domains_3models.png')
    X_tsne = plot_tsne_4domains_3models(X, sampled_models, sampled_domains, tsne_main_path)
    
    # 7. 绘制领域密度对比图
    density_path = os.path.join(OUTPUT_DIR, 'tsne_domain_density.png')
    plot_density_comparison(X_tsne, sampled_models, sampled_domains, density_path)
    
    # 8. 保存采样统计
    stats_path = os.path.join(OUTPUT_DIR, 'sampling_stats.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("Sampling Statistics\n")
        f.write("="*40 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value['sampled']}/{value['available']}\n")
    
    print(f"\n✅ 所有结果已保存至: {OUTPUT_DIR}")
    print(f"   - 主t-SNE图: tsne_4domains_3models.png")
    print(f"   - 领域密度图: tsne_domain_density.png")
    print(f"   - 采样统计: sampling_stats.txt")

if __name__ == "__main__":
    main()