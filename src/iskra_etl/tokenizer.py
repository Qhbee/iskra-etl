"""Tokenizer：中文块文本 jieba 分词 + 停用词过滤，供 ``tsvector`` / 混合检索离线链路。

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
from pathlib import Path

import jieba

_jieba_warmed: bool = False

# 停用词表：按语料直接改本集合即可（索引与 query 须保持一致）。
DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    {
        "的", "地", "得",
        "了", "着", "就", "都",
        "人", "自己", "你", "我", "他", "她", "它", "我们", "你们", "他们", "她们", "它们",
        "是", "不", "有", "没有",
        "一个",
        "在", "到", "去",
        "说", "要", "会",
        "和", "与", "及", "或", "等",
        "而", "但",
        "为", "以", "于", "之", "其", "所", "被", "把", "让", "给",
        "从", "对", "向", "由",
        "能", "可", "将", "已",
        "也", "又", "再", "还",
        "更", "最", "很", "非常",
        "已经", "正在", "刚才",
        "因为", "所以",
        "如果", "假设", "要是", "倘若",
        "虽然", "即使", "但是", "然而",
        "并且", "以及", "或者", "不是",
        "这", "那", "这个", "那个", "这些", "那些", "这么", "那么", "这样", "那样",
        "什么", "怎么", "怎样", "如何", "哪里", "哪个", "哪些", "多少", "为什么",
        "吗", "呢", "吧", "啊", "呀", "哦", "嗯", "哪",
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

    sw = DEFAULT_STOPWORDS if stopwords is None else stopwords
    tokens: list[str] = []
    for tok in jieba.cut_for_search(normalized_text, HMM=True):
        t = tok.strip()
        if not t or t in sw:
            continue
        tokens.append(t)
    return " ".join(tokens)
