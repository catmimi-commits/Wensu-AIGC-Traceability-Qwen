# 闻溯——AIGC新闻文本双轨制溯源与治理系统

两级检测系统：先做**主动水印检测**（命中即直接溯源到来源模型），未命中再回退到基于随机森林的**被动文风检测**（AI文本识别 + Qwen版本溯源）。

## 检测流程

```
用户提交文本
   │
   ├─【第一级】主动水印检测（KGW Z-score 跨密钥交叉验证）
   │     用每个模型的专属密钥复现「绿名单」，计算 Z-score
   │     ├─ 最高 Z-score ≥ 3.5 → 判定含水印，直接显示「检测到水印，来源是 XX 模型」
   │     └─ 最高 Z-score < 3.5 → 未检出明显水印，进入第二级
   │
   └─【第二级】被动文风检测（随机森林）
         输出 AI率 / 人类率、Qwen 版本分布概率、特征贡献度
```

> 名词说明：**KGW 水印**在生成每个 token 时，按「上一个 token + 模型密钥」把词表划出占比 γ=0.25 的**绿名单**并偏好输出其中的词；**Z-score** 衡量文本里绿 token 比例偏离 25% 的显著程度，越高越可能含水印；不同模型用不同**密钥**加水印，故哪个密钥算出的 Z-score 最高就溯源到哪个模型。详见 `网页检测逻辑/主动水印交叉验证.py`。

## 项目结构

```
源码/
├── backend/
│   ├── app.py                  # Flask主程序（两级检测路由）
│   ├── watermark_detector.py   # 第一级：主动水印检测（KGW Z-score）
│   ├── watermark_models/       # 水印检测所需的轻量分词器，仓库中单独将模型分出，运行时需自行调整位置
│   │   ├── Qwen1.5-0.5B-Instruct/
│   │   ├── Qwen2-0.5B-Instruct/
│   │   └── Qwen2.5-1.5B-Instruct/
│   ├── model/                  # 第二级：被动检测的随机森林模型
│   │   ├── ai_detector_binary.pkl
│   │   ├── ai_detector_tracking.pkl
│   │   ├── label_encoder_binary.pkl
│   │   ├── label_encoder_tracking.pkl
│   │   ├── feature_importance_binary.pkl
│   │   └── feature_importance_tracking.pkl
│   ├── feature/
│   │   ├── extractor.py        # 特征提取主函数
│   │   └── surface_features.py # 12项表层特征说明
│   └── requirements.txt        # Python依赖列表
├── frontend/
│   ├── static/
│   │   ├── main.js             # 交互脚本
│   │   ├── style.css           # 样式文件
│   │   └── exp1~5.png          # 实验图片
│   └── templates/
│       └── index.html          # 主页面
├── README.md
├── run.bat                     # Windows启动脚本
└── run.sh                      # Linux/Mac启动脚本
```

## 环境要求

- Python 3.8+
- 依赖见 `backend/requirements.txt`
- 其中 `torch` 与 `transformers` 用于第一级水印检测；**即使缺少这两个依赖，网页仍可正常运行**，只会自动跳过水印检测、直接进入第二级被动检测。

## 快速启动

### Windows
双击运行 `run.bat`，或在终端执行：
```bat
run.bat
```

### Linux / Mac
```bash
chmod +x run.sh
./run.sh
```

启动后浏览器访问：[http://localhost:5000](http://localhost:5000)

## 功能说明

**实时检测**：输入文本后先做主动水印检测；命中则直接显示水印来源模型与各密钥 Z-score，未命中再输出 AI率、人类概率、Qwen版本溯源分布及特征贡献度分析

**实验成果**：展示五项核心实验结果，包含数据集说明、跨域泛化、消融实验等

## 模型说明

| 文件 | 说明 |
|------|------|
| ai_detector_binary.pkl | 随机森林二分类器（Human / AI） |
| ai_detector_tracking.pkl | 随机森林多分类器（Qwen版本溯源） |
| label_encoder_*.pkl | 标签编码器 |
| feature_importance_*.pkl | 特征重要性（用于贡献度分析） |

> 模型基于约15,500条新闻文本训练，覆盖体育、科技、教育、时政四个领域。
