# -*- coding: utf-8 -*-
# 特征提取主函数
# 提取20维特征：12项表层特征 + 2项N-gram特征 + 6项单字频率特征
import re
import numpy as np
import jieba
import jieba.posseg as pseg


def extract_features(text):
    """
    提取20维文本特征
    :param text: 输入文本字符串
    :return: 长度为20的特征列表
    """
    if not isinstance(text, str) or not text.strip():
        return [0.0] * 20

    text = text.strip()
    char_count = len(text)
    words_with_pos = list(pseg.cut(text))
    words = [w for w, _ in words_with_pos]
    word_count = len(words) if words else 1

    f = []

    # 12项表层特征
    # 1. 平均词长（归一化）
    f.append((char_count / word_count) / 6.0)

    # 2. 标点密度
    punct = re.findall(r'[，。！？；：""''（）【】、]', text)
    f.append(len(punct) / char_count if char_count else 0.0)

    # 3. 数字密度
    digits = re.findall(r'\d+', text)
    f.append(len(''.join(digits)) / char_count if char_count else 0.0)

    # 4. 停用词比例
    stopwords = {'的','了','在','是','和','与','有','为','以','而','就','都','这','那','使','由'}
    f.append(sum(1 for w in words if w in stopwords) / word_count)

    # 5. 词汇丰富度
    f.append(len(set(words)) / word_count)

    # 6. 平均句长比例
    sentences = re.split(r'[。！？]', text)
    valid_sents = [s.strip() for s in sentences if s.strip()]
    sent_wc = [len(jieba.lcut(s)) for s in valid_sents] if valid_sents else []
    f.append((np.mean(sent_wc) if sent_wc else 0) / word_count)

    # 7. 功能词比例
    content_pos = {'a','v','n','nr','ns','nt','nz','vd','vn'}
    content_cnt = sum(1 for _, p in words_with_pos if p in content_pos)
    f.append(1 - content_cnt / word_count)

    # 8. 长句占比（>18词）
    f.append(sum(1 for c in sent_wc if c > 18) / len(valid_sents) if valid_sents else 0)

    # 9. 总结词比例
    summary_words = {'建议','本文','呼吁','总结','综上所述','总之','因此','由此可见'}
    f.append(sum(1 for w in words if w in summary_words) / word_count)

    # 10. 副词密度
    adv_words = {'很','更','非常','十分','极其','格外','稍微','几乎','往往','常常','大致','通常'}
    f.append(sum(1 for w in words if w in adv_words) / word_count)

    # 11. 连接词密度
    conj_words = {'因为','所以','但是','然而','而且','此外','同时','于是','据此','对此'}
    f.append(sum(1 for w in words if w in conj_words) / word_count)

    # 12. 单字功能词占比
    short_func = {'会','能','可','要','让','使','与','或','即'}
    f.append(sum(1 for w in words if len(w) == 1 and w in short_func) / word_count)

    # 2项N-gram特征
    qwen25_ng = {'，这','，而','，但','，更','，非常','，显著'}
    qwen2_ng  = {'，本文','，建议','，此外','，同时','，因此','，综上所述'}
    tng = {text[i]+text[i+1] for i in range(len(text)-1)}
    f.append(sum(1 for ng in qwen25_ng if ng in tng) / len(text) if text else 0)
    f.append(sum(1 for ng in qwen2_ng  if ng in tng) / len(text) if text else 0)

    # 6项单字频率特征
    for c in ['的','了','在','是','和','与']:
        f.append(text.count(c) / len(text) if text else 0)

    return f
