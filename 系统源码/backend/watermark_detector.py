# -*- coding: utf-8 -*-
"""
主动水印检测模块（KGW 水印 Z-score 跨密钥交叉验证）

逻辑来源：网页检测逻辑/主动水印交叉验证.py
----------------------------------------------------------------------
原理简述：
  KGW 水印在生成每个 token 时，会根据「上一个 token + 模型专属密钥 key」把整个词表
  划分出一份「绿名单」（占比 gamma=0.25），并让模型更倾向于输出绿名单里的词。
  检测时用同样的 key 复现每个位置的绿名单，统计实际落在绿名单里的 token 比例：
    - 无水印文本 ≈ gamma（25%），Z-score 接近 0；
    - 含水印文本 远高于 gamma，Z-score 显著为正。
  Z-score 把「绿 token 偏多」的程度标准化，Z ≥ 3.5（Z_THRESHOLD）即判定含水印。

  「跨密钥交叉验证」：每个 Qwen 版本用不同 key 加水印，因此用哪个 key 复现出的
  Z-score 最高，就说明文本来自哪个模型 —— 即可直接溯源到来源模型。

说明：水印检测只需要各模型的 tokenizer + config(vocab_size)，不需要模型权重
      (model.safetensors)，故 backend/watermark_models/ 下仅放了轻量分词器文件。
"""
import os
import math

# torch / transformers 为可选依赖：未安装时本模块自动禁用，
# 检测流程会自动回退到被动检测，不影响网页其余功能。
try:
    import torch
    from transformers import AutoTokenizer, AutoConfig
    _DEPS_OK = True
    _IMPORT_ERR = None
except Exception as _e:  # pragma: no cover
    _DEPS_OK = False
    _IMPORT_ERR = str(_e)

_BASE = os.path.dirname(os.path.abspath(__file__))

KGW_GAMMA   = 0.25   # 绿名单占词表比例（与生成端一致）
Z_THRESHOLD = 3.5    # 水印判定阈值

# 绿名单用 torch 随机数生成，CPU 与 GPU 的随机序列不同。
# ⚠️ 检测端 DEVICE 必须与「生成水印数据集时」所用设备一致，否则绿名单对不上、测不出水印。
# 默认自动检测；可用环境变量 WM_DEVICE=cpu / cuda 强制指定（无需改代码）：
#   set WM_DEVICE=cpu   （Windows）   export WM_DEVICE=cpu （Linux/Mac）
def _select_device():
    if not _DEPS_OK:
        return "cpu"
    want = os.environ.get("WM_DEVICE", "").strip().lower()
    if want == "cpu":
        return "cpu"
    if want == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"  # auto

DEVICE = _select_device()

# 分词器目录解析：优先用 backend/watermark_models/ 下的轻量副本，
# 找不到时回退到项目内的「网页检测逻辑」目录。
_LOCAL_DIR    = os.path.join(_BASE, "watermark_models")
_FALLBACK_DIR = os.path.abspath(os.path.join(_BASE, "..", "..", "..", "网页检测逻辑"))


def _resolve(sub_dir):
    local = os.path.join(_LOCAL_DIR, sub_dir)
    if os.path.exists(local):
        return local
    return os.path.join(_FALLBACK_DIR, sub_dir)


# 模型 -> { 分词器路径, 水印密钥 }，密钥与「主动水印交叉验证.py」保持一致
MODEL_CONFIGS = {
    "Qwen1.5-0.5B": {"path": _resolve("Qwen1.5-0.5B-Instruct"), "key": 62},
    "Qwen2-0.5B":   {"path": _resolve("Qwen2-0.5B-Instruct"),   "key": 52},
    "Qwen2.5-1.5B": {"path": _resolve("Qwen2.5-1.5B-Instruct"), "key": 42},
}

_tokenizers  = {}
_vocab_sizes = {}
_loaded      = False
_load_error  = None


def is_available():
    """依赖（torch + transformers）是否就绪。"""
    return _DEPS_OK


def get_status():
    """返回人类可读的模块状态，便于启动时打印与排查。"""
    if not _DEPS_OK:
        return f"不可用（缺少依赖：{_IMPORT_ERR}）"
    if _load_error:
        return f"不可用（{_load_error}）"
    if _loaded:
        return f"就绪，device={DEVICE}，阈值Z={Z_THRESHOLD}"
    return "尚未加载"


def load_watermark_models():
    """预加载各模型分词器与词表大小。返回 True 表示水印检测可用。"""
    global _loaded, _load_error
    if not _DEPS_OK:
        _load_error = f"缺少依赖（transformers/torch）：{_IMPORT_ERR}"
        return False
    if _loaded:
        return True
    try:
        for name, cfg in MODEL_CONFIGS.items():
            path = cfg["path"]
            if not os.path.exists(path):
                raise FileNotFoundError(f"未找到 {name} 的分词器目录：{path}")
            _tokenizers[name] = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            try:
                c = AutoConfig.from_pretrained(path, trust_remote_code=True)
                _vocab_sizes[name] = getattr(c, "vocab_size", _tokenizers[name].vocab_size)
            except Exception:
                _vocab_sizes[name] = _tokenizers[name].vocab_size
        _loaded = True
        _load_error = None
        return True
    except Exception as e:
        _load_error = str(e)
        return False


def calculate_kgw_z_score(text, tokenizer, real_vocab_size, gamma=0.25, hash_key=42, _cache=None):
    """复现生成端 LogitsProcessor 的 KGW Z-score 计算（与原脚本逐行对齐）。

    _cache：{prev_token: green_set} 的可选缓存。同一 (key, prev_token) 的绿名单完全相同，
            缓存后中文等高重复文本可显著提速，且不改变任何数值结果。
    """
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

        if _cache is not None and prev_token in _cache:
            green_tokens = _cache[prev_token]
        else:
            rng = torch.Generator(device=DEVICE)
            rng.manual_seed((hash_key * 10007 + prev_token) % (2**32 - 1))
            permutation = torch.randperm(real_vocab_size, generator=rng, device=DEVICE)
            green_tokens = set(permutation[:greenlist_size].tolist())
            if _cache is not None:
                _cache[prev_token] = green_tokens

        if curr_token in green_tokens:
            green_count += 1

    N = total_tokens - 1
    if N <= 0:
        return 0.0, total_tokens

    expected_green = gamma * N
    std_dev = math.sqrt(N * gamma * (1 - gamma))
    z_score = (green_count - expected_green) / std_dev if std_dev > 0 else 0.0
    return z_score, total_tokens


def detect_watermark(text):
    """对文本做跨密钥水印检测，返回结构化结果（供前端渲染）。

    返回字段：
      available   : 水印检测是否可用（依赖/分词器是否就绪）
      detected    : 是否检测到水印（最高 Z-score >= 阈值）
      threshold   : 判定阈值（3.5）
      best_model  : 命中时的来源模型名，否则 None
      max_z       : 最高 Z-score
      scores      : [{model, z, key, tokens}]，按 z 从高到低排序
      error       : 失败原因（若有）
    """
    base = {
        "available": False, "detected": False, "threshold": Z_THRESHOLD,
        "best_model": None, "max_z": 0.0, "scores": [], "error": None,
    }

    if not load_watermark_models():
        base["error"] = _load_error
        return base

    scores = []
    for name, cfg in MODEL_CONFIGS.items():
        cache = {}  # 每个 (模型, 密钥) 用独立缓存
        z, ntok = calculate_kgw_z_score(
            text=text,
            tokenizer=_tokenizers[name],
            real_vocab_size=_vocab_sizes[name],
            gamma=KGW_GAMMA,
            hash_key=cfg["key"],
            _cache=cache,
        )
        scores.append({
            "model": name,
            "z": round(float(z), 2),
            "key": cfg["key"],
            "tokens": int(ntok),
        })

    scores.sort(key=lambda s: s["z"], reverse=True)
    top = scores[0] if scores else None
    max_z = top["z"] if top else 0.0
    detected = bool(max_z >= Z_THRESHOLD)

    base.update({
        "available": True,
        "detected": detected,
        "best_model": top["model"] if (top and detected) else None,
        "max_z": max_z,
        "scores": scores,
    })
    return base


# 命令行自测：python watermark_detector.py "待检测文本"
if __name__ == "__main__":
    import sys, json
    sample = sys.argv[1] if len(sys.argv) > 1 else "这是一段用于测试水印检测流程的普通中文文本，用来观察 Z-score 的量级。"
    print("状态：", get_status())
    ok = load_watermark_models()
    print("加载：", get_status())
    if ok:
        print(json.dumps(detect_watermark(sample), ensure_ascii=False, indent=2))
