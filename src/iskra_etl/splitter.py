"""Splitter：LlamaIndex 读 Markdown → ``MarkdownNodeParser`` 按标题切段。不接 Embedding / VectorStore。

术语：「语料根」corpus_root 是磁盘上的目录；其下每个匹配的 ``**/index.md`` 是一「篇」文档
（LlamaIndex 的一个 ``Document``），切段后得到多条块与各篇内的 ``chunk_index``。
块边界由 Markdown 标题层级决定。正文若含 YAML frontmatter（``---``），切段前会先去除。
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from llama_index.core import Document, SimpleDirectoryReader

# include_prev_next_rel=False：不挂 prev/next 链；计划仅要线性 chunk_index。
from llama_index.core.node_parser import MarkdownNodeParser


@dataclass(frozen=True)
class ChunkRecord:
    """一条块：相对语料根的 posix 路径 + 篇内序号 + 纯文本。"""

    rel_path: str
    chunk_index: int
    chunk_text: str


def glob_index_paths(corpus_root: Path, glob_pattern: str = "**/index.md") -> list[Path]:
    root = corpus_root.resolve()
    return sorted(root.glob(glob_pattern))


def _rel_path_for_document(doc_file_path: str, corpus_root: Path) -> str:
    p = Path(doc_file_path).resolve()
    try:
        rel = p.relative_to(corpus_root.resolve())
    except ValueError as exc:
        msg = f"文档路径不在语料根下：file_path={doc_file_path!s}, corpus_root={corpus_root!s}"
        raise ValueError(msg) from exc
    return rel.as_posix()


def _strip_md_yaml_frontmatter(text: str) -> str:
    """丢弃 Jekyll/GitBook 风格的 YAML frontmatter（首尾 ``---``），保留正文 Markdown。

    仅以「单独一行在 trim 后为 ``---``」为分界；不按完整 YAML 解析。
    """
    if not text:
        return text
    t = text.lstrip("\ufeff")
    lines = t.splitlines(keepends=True)
    if not lines:
        return text

    def _line_trimmed(line: str) -> str:
        return line.rstrip("\r\n").strip("\t ")

    if _line_trimmed(lines[0]) != "---":
        return text

    for i in range(1, len(lines)):
        if _line_trimmed(lines[i]) == "---":
            rest = "".join(lines[i + 1 :]).lstrip("\r\n\v\f\u2028\u2029 ")
            # 删掉 frontmatter 后若正文全无，占位避免下游空 Document 异常边行为
            return rest if rest.strip() else "\n"

    return text


def split_corpus_to_chunks(
    corpus_root: Path,
    *,
    glob_pattern: str = "**/index.md",
    paths: Sequence[Path] | None = None,
    header_path_separator: str = "/",
    on_start: Callable[[int], None] | None = None,
    on_document_start: Callable[[int, int, str], None] | None = None,
    on_document_done: Callable[[int, int, str, int], None] | None = None,
) -> Iterator[ChunkRecord]:
    """遍历语料根下匹配的 Markdown，逐篇切段；每篇 chunk_index 从 0 递增。

    按 ``MarkdownNodeParser``：在标题边界切分；元数据里可带标题路径（由 ``header_path_separator`` 连接）。

    「语料」侧常见 ``index.md`` 会带 YAML frontmatter（首尾 ``---``）；切段前先 **去掉**，不产出纯元数据块。

    :param paths: 若给出则不再 glob，用于调用方事先统计篇数等与 ``glob_pattern`` 一致的一组文件。
    :param on_start: 加载文档前调用，参数为匹配的文档篇数 ``total_documents``。
    :param on_document_start: 开始处理某一篇时 ``(total_documents, index_1based, rel_path)``（在切段、写出该篇各块之前）。
    :param on_document_done: 每篇切段结束后调用 ``(total_documents, index_1based, rel_path, chunks_in_document)``.
    """
    corpus_root = corpus_root.resolve()
    resolved = (
        sorted(Path(p).resolve() for p in paths) if paths is not None else glob_index_paths(corpus_root, glob_pattern)
    )
    if not resolved:
        return iter(())

    if on_start is not None:
        on_start(len(resolved))

    reader = SimpleDirectoryReader(input_files=[str(p) for p in resolved])
    docs = reader.load_data()

    md_parser = MarkdownNodeParser(
        include_metadata=True,
        include_prev_next_rel=False,
        header_path_separator=header_path_separator,
    )
    total_docs = len(docs)

    def _gen() -> Iterator[ChunkRecord]:
        for doc_index_1, doc in enumerate(docs, start=1):
            fp = doc.metadata.get("file_path")
            if not fp:
                raise KeyError("LlamaIndex Document 缺少 metadata['file_path']")
            rel = _rel_path_for_document(str(fp), corpus_root)
            if on_document_start is not None:
                on_document_start(total_docs, doc_index_1, rel)
            raw_text = doc.text or ""
            cleaned = _strip_md_yaml_frontmatter(raw_text)
            doc_to_parse = doc if cleaned == raw_text else Document(text=cleaned, metadata=dict(doc.metadata))
            nodes = md_parser.get_nodes_from_documents([doc_to_parse])
            for i, node in enumerate(nodes):
                yield ChunkRecord(
                    rel_path=rel,
                    chunk_index=i,
                    chunk_text=node.get_content(),
                )
            if on_document_done is not None:
                on_document_done(total_docs, doc_index_1, rel, len(nodes))

    return _gen()


def chunk_records_to_jsonl_lines(records: Iterable[ChunkRecord]) -> Iterator[str]:
    """JSON Lines，每行 {\"rel_path\",\"chunk_index\",\"chunk_text\"}。"""
    for r in records:
        yield json.dumps(
            {
                "rel_path": r.rel_path,
                "chunk_index": r.chunk_index,
                "chunk_text": r.chunk_text,
            },
            ensure_ascii=False,
        )


def write_chunk_jsonl(path: Path, records: Iterable[ChunkRecord]) -> int:
    """写入 JSONL；返回行数。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for line in chunk_records_to_jsonl_lines(records):
            f.write(line + "\n")
            n += 1
    return n
