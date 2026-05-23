"""Tokenizer：中文块文本 jieba 分词 + 停用词过滤，供 ``tsvector`` / 混合检索离线链路。

**停用词**：首次调用 :func:`resolve_stopwords` 时从 GitHub 拉取 `goto456/stopwords` 的 ``cn_stopwords.txt``
  写入 ``.cache/cn_stopwords.txt`` 供离线复用；删该文件可强制重新下载。
  并与代码内 :data:`EXTRA_STOPWORDS`合并。

**分词策略（索引与 query 配对，勿混用）**

- **离线 Document**（本模块 ETL 文档入库端）：``tokenize_for_search`` → ``jieba.cut_for_search``（搜索引擎模式，长词再切子词，提高召回）。
- **在线 Query**（iskra-engine 用户检索端）：``tokenize_for_query`` → ``jieba.cut``（精确模式；停用词表与入库相同）。

读 JSONL 用 :func:`load_chunk_texts_from_jsonl`（整表进内存，与 ``embed_chunks`` 读 JSONL 方式一致）。
``chunks.jsonl`` 第 i 行对应 ``chunks_tokenized.txt`` 第 i 行，再与 ``chunks_embeddings.npy`` 的 ``embeddings[i]`` 对齐。

读 tokenized 行、拼 ``tsvector`` 字面量见 :mod:`iskra_etl.exporter`（Parquet 组装阶段）。
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import jieba

_jieba_warmed: bool = False

# goto456 整理的中文停用词表。
CN_STOPWORDS_URL = "https://raw.githubusercontent.com/goto456/stopwords/master/cn_stopwords.txt"
CN_STOPWORDS_CACHE_PATH = Path(__file__).resolve().parents[2] / ".cache" / "cn_stopwords.txt"

# 语料分词侧追加 STOPWORDS（与 ``resolve_stopwords`` 合并；每类一行，范围见行尾注释）
EXTRA_STOPWORDS: frozenset[str] = frozenset(
    {
        "#", "##", "###", "####", "#####", "######", "---",
        "~", "`", "@", "#", "$", "%", "^", "_", "+", "-", "*", "/", "=", "&", "|", "\\",
        "(", ")", "[", "]", "{", "}", "<", ">",
        ",", ".", "!", "?", ":", ";", '"', "'", "′", "″", "‴", "‵", "‶", "‷", "°",
        "（", "）", "［", "］", "｛ ", "｝", "〈", "〉", "〔", "〕", "【", "】", "「", "」",
        "⦅ ", "⦆ ", "〚", "〛", " ⦃ ", "⦄", "《", "》", "〘", "〙", "〖", "〗", "『", "』",
        "“", "”", "‘", "’", "￥", "·", "•", "．", "～", "—", "－", "…",
        "＋", "－", "×", "÷", "＊", "／", "＝", "％", "＜", "＞", "≠", "≈", "±", "％",
        "img", "assets", "png", "jpg", "jpeg", "gif", "svg", "webp",
        "０", "１", "２", "３", "４", "５", "６", "７", "８", "９",
        "Ａ", "Ｂ", "Ｃ", "Ｄ", "Ｅ", "Ｆ", "Ｇ", "Ｈ", "Ｉ", "Ｊ", "Ｋ", "Ｌ", "Ｍ", "Ｎ", "Ｏ", "Ｐ", "Ｑ", "Ｒ", "Ｓ", "Ｔ", "Ｕ", "Ｖ", "Ｗ", "Ｘ", "Ｙ", "Ｚ",
        "ａ", "ｂ", "ｃ", "ｄ", "ｅ", "ｆ", "ｇ", "ｈ", "ｉ", "ｊ", "ｋ", "ｌ", "ｍ", "ｎ", "ｏ", "ｐ", "ｑ", "ｒ", "ｓ", "ｔ", "ｕ", "ｖ", "ｗ", "ｘ", "ｙ", "ｚ",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "à", "á", "â", "ã", "ä", "å", "æ", "ç", "è", "é", "ê", "ë", "ì", "í", "î", "ï", "ñ", "ò", "ó", "ô", "õ", "ö", "ø", "ù", "ú", "û", "ü", "ý", "ÿ", "œ",
        "À", "Á", "Â", "Ã", "Ä", "Å", "Æ", "Ç", "È", "É", "Ê", "Ë", "Ì", "Í", "Î", "Ï", "Ñ", "Ò", "Ó", "Ô", "Õ", "Ö", "Ø", "Ù", "Ú", "Û", "Ü", "Ý", "Ÿ", "Œ",
        "Α", "Β", "Γ", "Δ", "Ε", "Ζ", "Η", "Θ", "Ι", "Κ", "Λ", "Μ", "Ν", "Ξ", "Ο", "Π", "Ρ", "Σ", "Τ", "Υ", "Φ", "Χ", "Ψ", "Ω",
        "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π", "ρ", "σ", "ς", "τ", "υ", "φ", "χ", "ψ", "ω",
        "Ё", "А", "Б", "В", "Г", "Д", "Е", "Ж", "З", "И", "Й", "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "У", "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ы", "Ь", "Э", "Ю", "Я",
        "ё", "а", "б", "в", "г", "д", "е", "ж", "з", "и", "й", "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я",
        "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩", "⑪", "⑫", "⑬", "⑭", "⑮", "⑯", "⑰", "⑱", "⑲", "⑳",
        "Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ", "Ⅵ", "Ⅶ", "Ⅷ", "Ⅸ", "Ⅹ", "Ⅺ", "Ⅻ", "Ⅼ", "Ⅽ", "Ⅾ", "Ⅿ",
        "ⅰ", "ⅱ", "ⅲ", "ⅳ", "ⅴ", "ⅵ", "ⅶ", "ⅷ", "ⅸ", "ⅹ", "ⅺ", "ⅻ", "ⅼ", "ⅽ", "ⅾ", "ⅿ",
        "⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹",
        "₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉",
    }
)


def tokenize_jsonl_to_txt(
    jsonl_path: Path | str,
    txt_path: Path | str,
    *,
    max_chunks: int | None = None,
    stopwords: frozenset[str] | None = None,
    show_progress_bar: bool = False,
) -> int:
    """``chunks.jsonl`` → ``chunks_tokenized.txt``（读入全部 ``chunk_text`` 后分词写出）。"""
    texts = load_chunk_texts_from_jsonl(jsonl_path)
    if max_chunks is not None:
        texts = texts[:max_chunks]
    return write_chunks_tokenized_txt(
        txt_path,
        texts,
        stopwords=stopwords,
        show_progress_bar=show_progress_bar,
    )


def load_chunk_texts_from_jsonl(path: Path | str) -> list[str]:
    """读 ``split_chunks`` 产出的 JSONL，返回全部 ``chunk_text``（跳过空物理行）。

    行序与 ``embedder.iter_chunk_records_from_jsonl`` 一致；一次性装入内存，便于 ``len`` 作进度条 total。
    """
    texts: list[str] = []
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            texts.append(str(o["chunk_text"]))
    return texts


def write_chunks_tokenized_txt(
    path: Path | str,
    chunk_texts: list[str],
    *,
    stopwords: frozenset[str] | None = None,
    show_progress_bar: bool = False,
) -> int:
    """写 ``chunks_tokenized.txt``：每块一行，UTF-8，行序与 JSONL / ``.npy`` 对齐。"""
    if chunk_texts:
        warmup_jieba()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for text in _iter_with_progress(chunk_texts, show_progress_bar=show_progress_bar):
            line = tokenize_for_search(text, stopwords=stopwords)
            f.write(line)
            f.write("\n")
            n += 1
    return n


def warmup_jieba() -> None:
    """预加载 jieba 词典，避免 tqdm 进行中首次 ``cut_for_search`` 往 stdout 打日志冲断进度条。"""
    global _jieba_warmed
    if _jieba_warmed:
        return
    jieba.setLogLevel(logging.WARNING)
    jieba.initialize()
    _jieba_warmed = True


def _iter_with_progress(
    texts: list[str],
    *,
    show_progress_bar: bool,
) -> Iterator[str]:
    if not show_progress_bar:
        yield from texts
        return
    try:
        from tqdm.auto import tqdm
    except ImportError:
        yield from texts
        return
    yield from tqdm(
        texts,
        total=len(texts),
        desc="分词",
        unit="chunk",
        mininterval=0.2,
        dynamic_ncols=True,
    )


def tokenize_for_search(
    text: str,
    *,
    stopwords: frozenset[str] | None = None,
) -> str:
    """``jieba.cut_for_search`` → 去空白 token → 去停用词 → 空格拼接（供 ``simple`` tsvector）。"""
    # 切段文本中的换行/制表符压成空格，避免 tokenized 行文件出现多物理行。
    normalized_text = " ".join(text.split())
    if not normalized_text:
        return ""

    sw = resolve_stopwords() if stopwords is None else stopwords
    tokens: list[str] = []
    for tok in jieba.cut_for_search(normalized_text, HMM=True):
        t = tok.strip()
        if not t or t in sw:
            continue
        tokens.append(t)
    return " ".join(tokens)


# tokenize_for_search() 每处理一块都会调一次 resolve_stopwords()
# @lru_cache(maxsize=1) 可以把停用词集合的计算结果缓存起来，避免每次都重新加载。（最多保留 1 组 缓存条目，带参数的函数，maxsize 才需要更大。）
@lru_cache(maxsize=1)
def resolve_stopwords() -> frozenset[str]:
    """停用词集合：GitHub 拉取 + ``.cache/cn_stopwords.txt`` 缓存（进程内 ``lru_cache``）。"""
    _get_and_cache_stopwords()
    with CN_STOPWORDS_CACHE_PATH.open(encoding="utf-8") as f:
        lines = f.readlines()
        words: set[str] = set()
        for line in lines:
            word = line.strip()
            if word:
                words.add(word)
        words |= EXTRA_STOPWORDS
        return frozenset(words)


def _get_and_cache_stopwords() -> None:
    """本地 ``CN_STOPWORDS_CACHE_PATH`` 已缓存 cn_stopwords.txt 则跳过；否则从 GitHub 下载文件到该路径。"""
    if CN_STOPWORDS_CACHE_PATH.is_file():
        return
    CN_STOPWORDS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(CN_STOPWORDS_URL, CN_STOPWORDS_CACHE_PATH)
    except URLError as exc:
        msg = f"无法从 GitHub 下载该中文停用词表: {CN_STOPWORDS_URL}"
        raise OSError(msg) from exc
