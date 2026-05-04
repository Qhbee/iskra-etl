"""Exporter：离线生成带显式 ``document_id`` 的 ``documents.parquet`` / ``chunks.parquet``。

``document_id`` 与语料下 ``glob`` 命中文件的 **排序顺序** 一致（默认与 :func:`iskra_etl.splitter.glob_index_paths` 相同），
从 ``id_start`` 起连续递增，供服务器 ``COPY`` 时插入 ``document.id``，并在 ``chunk`` 侧直接使用同一 ``document_id``。

**chunks.parquet** 推荐由 **``chunks.jsonl`` + 行对齐的 ``chunks_embeddings.npy``** 组装（不在此模块加载 ST）；
:func:`write_chunks_parquet_from_jsonl_and_npy` 负责校验行数与维数。
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from iskra_etl.embedder import iter_chunk_records_from_jsonl
from iskra_etl.splitter import (
    ChunkRecord,
    glob_index_paths,
    normalize_newlines,
    _strip_md_yaml_frontmatter,
)


def load_chunk_embeddings_npy(path: Path | str, *, mmap_mode: str | None = "r") -> np.ndarray:
    """加载与 ``chunks.jsonl`` **行对齐**的向量矩阵（``float32``、``(N, D)``）。

    默认 ``mmap_mode='r'``，大文件可减少 resident 内存；需要可写副本时传 ``mmap_mode=None``。
    """
    return np.load(path, mmap_mode=mmap_mode)


def default_document_id_start() -> int:
    raw = os.environ.get("ISKRA_DOCUMENT_ID_START", "").strip()
    return int(raw) if raw else 1


def document_id_by_rel_path(
    corpus_root: Path,
    *,
    glob_pattern: str = "**/index.md",
    paths: Sequence[Path] | None = None,
    id_start: int | None = None,
) -> dict[str, int]:
    """``rel_path``（posix）→ ``document_id``；顺序 = ``paths`` 或 ``glob`` 排序后的篇序。"""
    start = id_start if id_start is not None else default_document_id_start()
    root = corpus_root.resolve()
    resolved = (
        sorted(Path(p).resolve() for p in paths)
        if paths is not None
        else glob_index_paths(root, glob_pattern)
    )
    out: dict[str, int] = {}
    for i, p in enumerate(resolved):
        rel = Path(p).resolve().relative_to(root).as_posix()
        out[rel] = start + i
    return out


def _extract_frontmatter_block(raw: str) -> str | None:
    """仅识别「首行 ``---`` … 闭合 ``---``」之间的原始文本；不调用完整 YAML 解析。"""
    if not raw:
        return None
    t = raw.lstrip("\ufeff")
    lines = t.splitlines()
    if not lines:
        return None

    def _trim(line: str) -> str:
        return line.strip()

    if _trim(lines[0]) != "---":
        return None
    for i in range(1, len(lines)):
        if _trim(lines[i]) == "---":
            return "\n".join(lines[1:i])
    return None


def _scalar_yaml_value(raw: str) -> str:
    s = raw.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s


def title_book_from_raw_markdown(raw: str) -> tuple[str | None, str | None]:
    """从 frontmatter 里取 ``title`` / ``book`` 顶格键（大小写不敏感键名）。"""
    block = _extract_frontmatter_block(raw)
    if not block:
        return None, None
    title: str | None = None
    book: str | None = None
    for line in block.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        low = s.lower()
        if low.startswith("title:"):
            title = _scalar_yaml_value(s.split(":", 1)[1])
        elif low.startswith("book:"):
            book = _scalar_yaml_value(s.split(":", 1)[1])
    return title, book


def document_row_for_path(
    corpus_root: Path,
    rel_path: str,
    document_id: int,
) -> dict[str, object]:
    """单行 ``documents.parquet`` 逻辑行（``dict``）。"""
    root = corpus_root.resolve()
    file_path = root.joinpath(*rel_path.split("/"))
    if not file_path.is_file():
        msg = f"语料文件不存在: {file_path}"
        raise FileNotFoundError(msg)
    raw = file_path.read_text(encoding="utf-8")
    full_text = normalize_newlines(raw)
    body = normalize_newlines(_strip_md_yaml_frontmatter(raw))
    content_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
    title, book = title_book_from_raw_markdown(raw)
    return {
        "id": int(document_id),
        "rel_path": rel_path,
        "title": title,
        "book": book,
        "full_text": full_text,
        "content_sha256": content_sha256,
    }


def build_documents_rows(
    corpus_root: Path,
    id_by_rel: dict[str, int],
) -> list[dict[str, object]]:
    """按 ``document_id`` 升序生成行，便于与服务器导入顺序一致。"""
    rows: list[dict[str, object]] = []
    for rel_path, doc_id in sorted(id_by_rel.items(), key=lambda x: x[1]):
        rows.append(document_row_for_path(corpus_root, rel_path, doc_id))
    return rows


def write_documents_parquet(rows: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    cols = ["id", "rel_path", "title", "book", "full_text", "content_sha256"]
    df = df[cols]
    df.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def write_chunks_parquet(
    *,
    document_id_by_rel: dict[str, int],
    records: Sequence[ChunkRecord],
    embeddings: np.ndarray,
    path: Path,
    expect_dim: int | None = None,
) -> None:
    if len(records) != len(embeddings):
        msg = f"块数 {len(records)} 与向量行数 {len(embeddings)} 不一致"
        raise ValueError(msg)
    if embeddings.ndim != 2:
        msg = f"embeddings 期望二维，shape={embeddings.shape}"
        raise ValueError(msg)
    dim = int(embeddings.shape[1])
    if expect_dim is not None and dim != expect_dim:
        msg = f"向量维数 {dim} 与期望 {expect_dim} 不一致"
        raise ValueError(msg)

    doc_ids: list[int] = []
    missing: list[str] = []
    for r in records:
        did = document_id_by_rel.get(r.rel_path)
        if did is None:
            missing.append(r.rel_path)
        else:
            doc_ids.append(int(did))
    if missing:
        sample = ", ".join(sorted(set(missing))[:5])
        msg = f"{len(set(missing))} 个 rel_path 不在 document_id 映射中（示例）: {sample}"
        raise KeyError(msg)

    emb32 = np.asarray(embeddings, dtype=np.float32)
    emb_list = [row.tolist() for row in emb32]
    df = pd.DataFrame(
        {
            "document_id": doc_ids,
            "rel_path": [r.rel_path for r in records],
            "chunk_index": [int(r.chunk_index) for r in records],
            "text": [r.chunk_text for r in records],
            "embedding": emb_list,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def load_chunk_records_from_jsonl(
    path: Path | str,
    *,
    max_chunks: int | None = None,
) -> list[ChunkRecord]:
    """读 JSONL 为 :class:`ChunkRecord` 列表（顺序与 ``.npy`` 行严格一致）。"""
    records: list[ChunkRecord] = []
    for i, r in enumerate(iter_chunk_records_from_jsonl(path)):
        if max_chunks is not None and i >= max_chunks:
            break
        records.append(r)
    return records


def write_chunks_parquet_from_jsonl_and_npy(
    *,
    chunks_jsonl: Path | str,
    embeddings_npy: Path | str,
    document_id_by_rel: dict[str, int],
    path: Path,
    expect_dim: int | None = None,
    max_chunks: int | None = None,
    mmap_npy: bool = True,
) -> tuple[int, int]:
    """读 **行对齐** 的 JSONL + ``float32`` ``(N, D)`` 向量文件，写 ``chunks.parquet``。

    :return: ``(块数 N, 维数 D)``
    """
    records = load_chunk_records_from_jsonl(chunks_jsonl, max_chunks=max_chunks)
    mmap_mode = "r" if mmap_npy else None
    embeddings = load_chunk_embeddings_npy(embeddings_npy, mmap_mode=mmap_mode)

    n_rec = len(records)
    n_emb = int(embeddings.shape[0])
    if n_rec != n_emb:
        msg = (
            f"JSONL 块数 {n_rec} 与 embeddings 行数 {n_emb} 不一致；"
            "须用同次 embed 产物或相同 --max-chunks。"
        )
        raise ValueError(msg)

    write_chunks_parquet(
        document_id_by_rel=document_id_by_rel,
        records=records,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        path=path,
        expect_dim=expect_dim,
    )
    dim = int(embeddings.shape[1]) if embeddings.size else 0
    return n_rec, dim
