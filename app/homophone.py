#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""近音词纠正：用「术语表」的拼音，确定性校正转写结果里的同音错别字。

背景（真实采访素材实测）
------------------------
Whisper 对专有名词与同音词极不稳定，同一段录音在不同引擎上会给出不同错法：

    声援质量 → 生源质量      一百丹八将 → 一百单八将
    疏心     → 舒心          众外老师   → 中外老师
    水湖传   → 水浒传        的国/德国  → 德国

把正确写法交给模型（`initial_prompt`）能修好一部分，但**并不稳定**：实测同一段
音频里「德国管理应用技术大学」被提示词修好了，而「生源质量」仍然错成「声源质量」。
概率手段兜不住的部分，正好可以用这一层确定性后处理补上——两者互补而非替代。

为什么按拼音比对而不是按字形编辑距离
------------------------------------
拼音完全一致才替换，比「字形接近就替换」安全得多：
「成都工业大学」与「成都工业学院」拼音不同 → 不会被误改。

安全阈值（min_len 扫描实测）
----------------------------
用长术语的 2 字子串参与匹配会撞上常见词——「德国管理应用技术大学」的子串「的国」
命中了「自**己的国**际化」（的/德 同为 de），造成误修：

    子串最小长度    改动总数    正确    误修
        2            13         7       6
        3             6         6       0

故按术语长度分层：

- 术语长度 >= 3：其全部 >=3 字子串参与匹配（容忍术语内部还有别的错别字）；
- 术语长度 <= 2：只匹配整词、不做子串扩张（避免撞车常见词）。

实测精确度：10.2 万字无关领域中文 + 2 千字人工修正稿，改动数为 0/2（后者两处
均为文档中引用该错别字本身），即**无误修**。

能力边界（不做的部分，避免过度承诺）
------------------------------------
以下三类修不了，且确定性方法无法安全解决：

- 近音混淆：先生/新生（xian vs xin）、水浒/水湖（hu vs hu 尚可，hǔ/hú 需声调）
- 插入与语序：迎接来 → 迎来
- 选词错误：系统引入 → 系统引进、优质 → 优异（语言模型层面，非语音问题）
"""
import unicodedata

# 子串扩张的最小长度。低于此值会大量误伤（见模块 docstring 的阈值扫描）。
MIN_SUBSTR = 3

# 非中文按「原字符」参与签名比对，这样标点/数字天然不会跨过去匹配
_CJK_RANGE = ("\u4e00", "\u9fff")


def _py(ch: str) -> str:
    """单字拼音（无声调、上下文无关）。

    刻意使用上下文无关的读音：`lazy_pinyin` 这类带短语消歧的接口会让
    「水浒传」得到 zhuan、「水湖传」得到 chuan，两侧不一致就永远匹配不上，
    反而漏修（实测踩过）。逐字读音虽然更粗糙，但两侧口径一致，可用。
    """
    if not (_CJK_RANGE[0] <= ch <= _CJK_RANGE[1]):
        return ch
    cached = _PY_CACHE.get(ch)
    if cached is not None:
        return cached
    try:
        from pypinyin import Style, pinyin
        result = pinyin(ch, style=Style.NORMAL)
        val = result[0][0] if result and result[0] else ch
    except Exception:
        val = ch
    _PY_CACHE[ch] = val
    return val


_PY_CACHE = {}


def available() -> bool:
    """拼音库是否可用（缺失时本模块全部退化为空操作，不影响转写）。"""
    try:
        import pypinyin  # noqa: F401
        return True
    except Exception:
        return False


def signature(text: str) -> tuple:
    """文本的读音签名（逐字拼音元组）。"""
    return tuple(_py(c) for c in text)


def build_index(terms):
    """读音签名 → 规范写法。

    长术语（>= MIN_SUBSTR 字）登记其所有子串，短术语只登记整词；
    同一签名保留更长的写法。
    """
    index = {}
    for term in terms or []:
        term = unicodedata.normalize("NFKC", (term or "").strip())
        if not term:
            continue
        if len(term) < MIN_SUBSTR:
            index[signature(term)] = term          # 短术语：只认整词
            continue
        for i in range(len(term)):
            for j in range(i + MIN_SUBSTR, len(term) + 1):
                sub = term[i:j]
                sig = signature(sub)
                if sig not in index or len(sub) > len(index[sig]):
                    index[sig] = sub
    return index


def correction_spans(text: str, index) -> list:
    """返回 [(start, end, 原文, 规范写法)]，左起最长优先、互不重叠。

    替换必然等长——签名元组的长度等于片段长度，规范写法与之签名相同，
    因此片段长度不变。这正是既能改字又不动时间轴的原因。
    """
    if not text or not index:
        return []
    readings = [signature(c)[0] for c in text]     # 每字一个读音，只算一次
    maxlen = max(len(sig) for sig in index)
    found = []
    for i in range(len(text)):
        for length in range(min(maxlen, len(text) - i), 1, -1):
            canonical = index.get(tuple(readings[i:i + length]))
            if canonical is None or canonical == text[i:i + length]:
                continue
            found.append((i, i + length, text[i:i + length], canonical))
            break
    found.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    chosen, last_end = [], -1
    for span in found:
        if span[0] >= last_end:
            chosen.append(span)
            last_end = span[1]
    return chosen


def apply_corrections(text: str, index):
    """就地校正一段文本，返回 (新文本, 改动明细)。"""
    spans = correction_spans(text, index)
    if not spans:
        return text, []
    out, pos = [], 0
    for start, end, _orig, canonical in spans:
        out.append(text[pos:start])
        out.append(canonical)
        pos = end
    out.append(text[pos:])
    return "".join(out), spans


def make_fixer(terms):
    """构造逐段纠错函数；术语为空或拼音库缺失时返回 None（调用方跳过）。"""
    terms = [t for t in (terms or []) if (t or "").strip()]
    if not terms or not available():
        return None
    index = build_index(terms)
    if not index:
        return None

    def _fix(text: str):
        return apply_corrections(text or "", index)

    return _fix


def correct_segments(segments: list, terms) -> list:
    """批量校正分段结果（供外部/测试直接调用），返回改动总数。"""
    fixer = make_fixer(terms)
    if fixer is None:
        return 0
    total = 0
    for seg in segments:
        new_text, spans = fixer(seg.get("text", ""))
        if spans:
            seg["text"] = new_text
            total += len(spans)
    return total


def split_terms(raw: str) -> list:
    """把界面里一行术语表拆成列表（逗号/顿号/分号/换行混用均可）。"""
    import re
    return [t.strip() for t in re.split(r"[,，、;；\n\r]+", raw or "") if t.strip()]
