# -*- coding: utf-8 -*-
# Flask后端主程序
# 两级检测逻辑：
#   第一级 主动水印检测：KGW 水印 Z-score 跨密钥交叉验证，命中则直接溯源到来源模型
#   第二级 被动文风检测：人机二分类 + Qwen版本溯源 + 特征贡献度分析（仅在未检出水印时执行）
from flask import Flask, request, jsonify, render_template
import os, re
import numpy as np
import joblib
import jieba
import jieba.posseg as pseg

# 主动水印检测模块（torch/transformers 为可选依赖，缺失时自动回退到被动检测）
import watermark_detector

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend', 'templates'),
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend', 'static'))

# 路径配置
_BASE     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_BASE, "model")

# 示例文本
SAMPLE_TEXTS = {
    "体育": "记者从中国武术协会获悉，3月21日上午，首个国际太极拳日太极拳展演活动主会场在福建武夷山举行。国家体育总局局长、党组书记、国际武术联合会主席高志丹出席并宣布活动开幕。此次活动以健康生活、太极同行为主题，采用主会场、分会场联动形式，与湖北十堰、河南焦作、河北邯郸三地同步开展，共庆联合国教科文组织设立的首个以武术项目命名的国际日。2025年11月5日，联合国教科文组织第43届大会正式将每年3月21日设立为国际太极拳日，标志着太极拳的文化价值与健康理念获得全球广泛认同，成为推动文明交流互鉴、践行全球文明倡议的生动实践。",
    "科技": "2026年3月10日，在法国巴黎举行的第二届核能峰会上，中国宣布加入由22个国家在第28届联合国气候变化大会上共同发起的三倍核能宣言，这一举措为促进全球核能可持续发展和能源绿色低碳转型注入强劲动力。中国国家原子能机构主任单忠德出席峰会三倍核能计划圆桌讨论，并代表中方宣布加入三倍核能宣言。单忠德表示，实现2050年全球核能增至三倍的目标，需要各国团结协作、共同努力。作为巴黎协定的重要推动者和积极践行者，中国信念坚定、行动果决，始终将核能作为保障能源安全和应对气候变化的重要途径。",
    "教育": "期末将至，考试再次成为各级学校学生的关注焦点。记者从教育部了解到，周周测、月月考的现象将成为过去。教育部办公厅出台关于进一步加强中小学日常考试管理的通知，对各学段考试频次做出明确规定。通知要求：小学一、二年级不进行纸笔考试，义务教育其他年级由学校每学期组织一次期末考试，初中年级从不同学科的实际出发，可适当安排一次期中考试。普通高中学校要严格控制考试次数。同时，严禁面向小学各年级和初高中非毕业年级组织区域性或跨校际的考试，初高中毕业年级可在总复习阶段组织1至2次模拟考试。",
    "时政": "全国政协主席王沪宁20日在京会见美国哈佛大学教授艾利森。王沪宁表示，习近平主席同特朗普总统保持着良好交往，为中美关系改善发展提供了重要战略引领。今年是中国十五五开局之年，中国将以自身发展的确定性应对风云变幻的国际形势。台湾问题是中国核心利益中的核心。中美双方应当加强对话沟通，妥善管控分歧，拓展务实合作，走出一条正确相处之道，为世界注入确定性和正能量。艾利森表示，当前国际和平与秩序正遭受严重侵蚀，美中找到正确相处之道对世界很重要，希望美中双方妥善处理台湾等问题，确保两国关系稳定发展。"
}

# 全局模型变量
clf_binary   = None
le_binary    = None
clf_tracking = None
le_tracking  = None
fi_binary    = None
fi_tracking  = None

SURFACE_FEATURE_NAMES = [
    '平均词长', '标点密度', '数字密度', '停用词比例',
    '词汇丰富度', '平均句长', '功能词比例', '长句占比',
    '总结词比例', '副词密度', '连接词密度', '单字功能词占比'
]

# 加载模型
def load_models():
    global clf_binary, le_binary, clf_tracking, le_tracking, fi_binary, fi_tracking
    print(f"MODEL_DIR = {os.path.abspath(MODEL_DIR)}")
    try:
        clf_binary   = joblib.load(os.path.join(MODEL_DIR, "ai_detector_binary.pkl"))
        le_binary    = joblib.load(os.path.join(MODEL_DIR, "label_encoder_binary.pkl"))
        clf_tracking = joblib.load(os.path.join(MODEL_DIR, "ai_detector_tracking.pkl"))
        le_tracking  = joblib.load(os.path.join(MODEL_DIR, "label_encoder_tracking.pkl"))
        # 加载特征重要性
        fi_path_b = os.path.join(MODEL_DIR, "feature_importance_binary.pkl")
        fi_path_t = os.path.join(MODEL_DIR, "feature_importance_tracking.pkl")
        def _load_fi(path, clf):
            if os.path.exists(path):
                raw = joblib.load(path)
                if isinstance(raw, dict):
                    return np.array(list(raw.values()))
                return np.array(raw)
            if hasattr(clf, 'feature_importances_'):
                return clf.feature_importances_
            return None
        fi_binary   = _load_fi(fi_path_b, clf_binary)
        fi_tracking = _load_fi(fi_path_t, clf_tracking)
        print("模型加载成功")
        return True
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False

# 特征提取（20维，与训练保持一致）
def extract_features(text):
    if not isinstance(text, str) or not text.strip():
        return [0.0] * 20
    text = text.strip()
    char_count = len(text)
    words_with_pos = list(pseg.cut(text))
    words = [w for w, _ in words_with_pos]
    word_count = len(words) if words else 1

    f = []
    # 平均词长（归一化）
    f.append((char_count / word_count) / 6.0)
    # 标点密度
    punct = re.findall(r'[，。！？；：""''（）【】、]', text)
    f.append(len(punct) / char_count if char_count else 0.0)
    # 数字密度
    digits = re.findall(r'\d+', text)
    f.append(len(''.join(digits)) / char_count if char_count else 0.0)
    # 停用词比例
    stopwords = {'的','了','在','是','和','与','有','为','以','而','就','都','这','那','使','由'}
    f.append(sum(1 for w in words if w in stopwords) / word_count)
    # 词汇丰富度
    f.append(len(set(words)) / word_count)
    # 平均句长比例
    sentences = re.split(r'[。！？]', text)
    valid_sents = [s.strip() for s in sentences if s.strip()]
    sent_wc = [len(jieba.lcut(s)) for s in valid_sents] if valid_sents else []
    f.append((np.mean(sent_wc) if sent_wc else 0) / word_count)
    # 功能词比例
    content_pos = {'a','v','n','nr','ns','nt','nz','vd','vn'}
    content_cnt = sum(1 for _, p in words_with_pos if p in content_pos)
    f.append(1 - content_cnt / word_count)
    # 长句占比（超过18词）
    f.append(sum(1 for c in sent_wc if c > 18) / len(valid_sents) if valid_sents else 0)
    # 总结词比例
    summary_words = {'建议','本文','呼吁','总结','综上所述','总之','因此','由此可见'}
    f.append(sum(1 for w in words if w in summary_words) / word_count)
    # 副词密度
    adv_words = {'很','更','非常','十分','极其','格外','稍微','几乎','往往','常常','大致','通常'}
    f.append(sum(1 for w in words if w in adv_words) / word_count)
    # 连接词密度
    conj_words = {'因为','所以','但是','然而','而且','此外','同时','于是','据此','对此'}
    f.append(sum(1 for w in words if w in conj_words) / word_count)
    # 单字功能词占比
    short_func = {'会','能','可','要','让','使','与','或','即'}
    f.append(sum(1 for w in words if len(w) == 1 and w in short_func) / word_count)
    # N-gram特征
    qwen25_ng = {'，这','，而','，但','，更','，非常','，显著'}
    qwen2_ng  = {'，本文','，建议','，此外','，同时','，因此','，综上所述'}
    tng = {text[i]+text[i+1] for i in range(len(text)-1)}
    f.append(sum(1 for ng in qwen25_ng if ng in tng) / len(text) if text else 0)
    f.append(sum(1 for ng in qwen2_ng  if ng in tng) / len(text) if text else 0)
    # 单字频率特征
    for c in ['的','了','在','是','和','与']:
        f.append(text.count(c) / len(text) if text else 0)
    return f

# 特征贡献度计算
# 20维特征分组：表层(0-11)、N-gram(12-13)、单字频率(14-19，近似TF-IDF)
def compute_contribution(features_arr, fi):
    # 基于特征重要性计算三维贡献度
    if fi is None or not hasattr(fi, '__len__') or len(fi) < 20:
        return {'表层特征': 33.3, 'TF-IDF': 33.3, 'N-gram': 33.3}
    feat = np.abs(np.array(features_arr[:20]) * fi[:20])
    surface = float(np.sum(feat[0:12]))
    ngram   = float(np.sum(feat[12:14]))
    tfidf   = float(np.sum(feat[14:20]))
    total   = surface + ngram + tfidf
    if total == 0:
        return {'表层特征': 33.3, 'TF-IDF': 33.3, 'N-gram': 33.3}
    return {
        '表层特征': round(surface / total * 100, 1),
        'TF-IDF':   round(tfidf   / total * 100, 1),
        'N-gram':   round(ngram   / total * 100, 1),
    }

def compute_surface_detail(features_arr, fi):
    # 12项表层特征各自贡献度，归一化到100%
    if fi is None or not hasattr(fi, '__len__') or len(fi) < 12:
        return {n: round(100/12, 1) for n in SURFACE_FEATURE_NAMES}
    feat = np.abs(np.array(features_arr[0:12]) * fi[0:12])
    total = float(np.sum(feat))
    if total == 0:
        return {n: round(100/12, 1) for n in SURFACE_FEATURE_NAMES}
    return {n: round(float(feat[i]) / total * 100, 1) for i, n in enumerate(SURFACE_FEATURE_NAMES)}

# 预测主函数
def predict(text):
    feat = extract_features(text)
    feat_arr = np.array(feat).reshape(1, -1)

    # 第一层：Human/AI二分类
    pred_bin   = clf_binary.predict(feat_arr)[0]
    label      = le_binary.inverse_transform([pred_bin])[0]
    proba_bin  = clf_binary.predict_proba(feat_arr)[0]
    classes_b  = list(le_binary.classes_)
    ai_prob    = float(proba_bin[classes_b.index('AI')])    if 'AI'    in classes_b else 0.0
    human_prob = float(proba_bin[classes_b.index('Human')]) if 'Human' in classes_b else 0.0

    # 第二层：Qwen版本溯源
    pred_track      = clf_tracking.predict(feat_arr)[0]
    best_track      = le_tracking.inverse_transform([pred_track])[0]
    proba_track     = clf_tracking.predict_proba(feat_arr)[0]
    track_raw       = {cls: round(float(proba_track[i])*100, 2)
                       for i, cls in enumerate(le_tracking.classes_)}
    total_t = sum(track_raw.values())
    track_norm = {k: round(v/total_t*100, 2) for k, v in track_raw.items()} if total_t else track_raw

    # 特征贡献度计算
    # 清理溯源标签，去掉参数后缀只保留版本号
    def clean_label(name):
        for prefix in ['Qwen2.5', 'Qwen2', 'Qwen1.5']:
            if name.startswith(prefix):
                return prefix
        return name

    track_norm  = {clean_label(k): v for k, v in track_norm.items()}
    best_track  = clean_label(best_track)
    fi_used = fi_tracking if fi_tracking is not None else fi_binary
    dim_contrib    = compute_contribution(feat, fi_used)
    surface_detail = compute_surface_detail(feat, fi_used)

    return {
        "label":          label,
        "ai_prob":        round(ai_prob * 100, 2),
        "human_prob":     round(human_prob * 100, 2),
        "confidence":     round(max(ai_prob, human_prob) * 100, 2),
        "tracking": {
            "best":  best_track,
            "probs": track_norm
        },
        "contribution":     dim_contrib,
        "surface_detail":   surface_detail
    }

# 路由
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "请输入文本"}), 400
    if clf_binary is None:
        return jsonify({"error": "模型未加载，请检查model路径"}), 500
    try:
        # ---- 第一级：主动水印检测 ----
        wm = watermark_detector.detect_watermark(text)
        if wm.get("detected"):
            # 检测到水印 → 直接溯源，跳过被动检测
            return jsonify({"stage": "watermark", "watermark": wm})

        # ---- 第二级：未检出水印 → 被动文风检测 ----
        result = predict(text)
        result["stage"] = "passive"
        result["watermark"] = wm
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/sample/<domain>")
def sample(domain):
    text = SAMPLE_TEXTS.get(domain)
    if not text:
        return jsonify({"error": "未知领域"}), 404
    return jsonify({"text": text})

# 启动
if __name__ == "__main__":
    load_models()
    # 预加载水印检测分词器（失败不影响启动，仅退化为被动检测）
    watermark_detector.load_watermark_models()
    print(f"水印检测模块：{watermark_detector.get_status()}")
    app.run(debug=True, port=5000)
