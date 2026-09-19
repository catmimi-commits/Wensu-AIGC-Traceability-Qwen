# Wensu-AIGC-Traceability-Qwen
本项目面向中文新闻概括场景的 AIGC 双轨制溯源，相关实验融合正交 KGW 主动水印与轻量化浅层文风指纹，辅助实现大模型代际版本（Qwen1.5/2/2.5）精准归因与全网巡查。 A dual-track AIGC attribution system for Chinese news summarization. Integrates orthogonal KGW watermarks and lightweight stylistic fingerprints for precise LLM version tracking (Qwen1.5/2/2.5) and passive screening.
# Wensu-AIGC-Traceability-Qwen

## 项目简介
- 场景：中文新闻短文本（<300字）概括性改写
- 目标：从“是否AI生成”到“具体哪一代Qwen模型生成”
- 方法：主动水印（KGW+正交密钥）+ 被动指纹（浅层统计+TF-IDF）
- 本仓库开放：核心实验代码、检测模型、采样数据集

## 仓库结构
- `watermark_models/`：Qwen1.5/2/2.5水印检测所需分词器与配置(完整模型可自行搜索Hugging Face进行下载)
- `四领域短文本数据集/`：体育、科技、教育、时政四领域采样数据（原新闻文本取材自THUCNews新闻数据集）
- `实验代码/`：消融、跨领域迁移、水印归因等实验脚本
- `LICENSE`：Apache-2.0
- `README.md`：本文件

