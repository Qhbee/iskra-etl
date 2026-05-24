"""组装 ``documents.parquet`` + ``chunks.parquet``。

``documents.parquet`` 由读原始语料生成；
``chunks.parquet`` 由 ``chunks.jsonl`` + 行对齐的 ``chunks_embeddings.npy`` + ``chunks_tokenized.txt`` 组装（**不**加载 Sentence-Transformers）。
切块文件由 ``split_chunks.py`` 写出，向量文件由 ``embed_chunks.py`` 写出，分词文件由 ``tokenize_chunks.py`` 写出；
切块的 .jsonl 与 向量的 .npy 与分词的 .txt 必须同一顺序、同一块数。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")


def main() -> None:
    from iskra_etl.exporter import (
        build_documents_rows,
        default_document_id_start,
        document_id_by_rel_path,
        write_chunks_parquet_from_jsonl_and_npy_and_txt,
        write_documents_parquet,
    )

    ap = argparse.ArgumentParser(
        description="导出 documents.parquet + chunks.parquet（chunks 来自 JSONL + 行对齐 .npy + 行对齐 .txt）",
    )
    ap.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="语料根（默认 ISKRA_CORPUS_ROOT 或 ../iskra-data）",
    )
    ap.add_argument(
        "--chunks-jsonl",
        type=Path,
        default=None,
        help="切块 JSONL（默认 ISKRA_CHUNK_JSONL 或 out/chunks.jsonl）",
    )
    ap.add_argument(
        "--embeddings-npy",
        type=Path,
        default=None,
        help="行对齐向量 .npy（默认 ISKRA_CHUNK_EMBEDDINGS_NPY 或 out/chunks_embeddings.npy）",
    )
    ap.add_argument(
        "--tokenized-txt",
        type=Path,
        default=None,
        help="行对齐分词 .txt（默认 ISKRA_CHUNK_TOKENIZED_TXT 或 out/chunks_tokenized.txt）",
    )
    ap.add_argument(
        "--documents-parquet",
        type=Path,
        default=None,
        help="输出 documents.parquet（默认 ISKRA_DOCUMENTS_PARQUET 或 out/documents.parquet）",
    )
    ap.add_argument(
        "--chunks-parquet",
        type=Path,
        default=None,
        help="输出 chunks.parquet（默认 ISKRA_CHUNKS_PARQUET 或 out/chunks.parquet）",
    )
    ap.add_argument(
        "--glob",
        dest="glob_pattern",
        default="**/index.md",
        help="与切块脚本一致的 glob",
    )
    ap.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="仅组装前 N 条（jsonl 只读前 N 行；须与生成 .npy 和 .txt 时所用 N 一致）",
    )
    ap.add_argument(
        "--skip-validate-dim",
        action="store_true",
        help="不校验 ISKRA_EMBED_DIM（默认会校验）",
    )
    ap.add_argument(
        "--quiet",
        "-q",
        action="store_true",
    )
    args = ap.parse_args()

    corpus = args.corpus_root
    if corpus is None:
        raw = os.environ.get("ISKRA_CORPUS_ROOT", "").strip()
        corpus = Path(raw) if raw else _ROOT.parent / "iskra-data"
    corpus = corpus.resolve()

    if not corpus.is_dir():
        print(f"语料根无效: {corpus}", file=sys.stderr)
        sys.exit(2)

    jsl = args.chunks_jsonl
    if jsl is None:
        raw_j = os.environ.get("ISKRA_CHUNK_JSONL", "").strip()
        jsl = Path(raw_j) if raw_j else _ROOT / "out" / "chunks.jsonl"
    jsl = jsl.resolve()
    if not jsl.is_file():
        print(f"chunks JSONL 不存在: {jsl}", file=sys.stderr)
        sys.exit(2)

    npy = args.embeddings_npy
    if npy is None:
        raw_n = os.environ.get("ISKRA_CHUNK_EMBEDDINGS_NPY", "").strip()
        npy = Path(raw_n) if raw_n else _ROOT / "out" / "chunks_embeddings.npy"
    npy = npy.resolve()
    if not npy.is_file():
        print(f"embeddings .npy 不存在: {npy}（请先运行 scripts/embed_chunks.py）", file=sys.stderr)
        sys.exit(2)

    txt = args.tokenized_txt
    if txt is None:
        raw_t = os.environ.get("ISKRA_CHUNK_TOKENIZED_TXT", "").strip()
        txt = Path(raw_t) if raw_t else _ROOT / "out" / "chunks_tokenized.txt"
    txt = txt.resolve()
    if not txt.is_file():
        print(f"chunks_tokenized.txt 不存在: {txt}（请先运行 scripts/tokenize_chunks.py）", file=sys.stderr)
        sys.exit(2)

    pq_doc = args.documents_parquet
    if pq_doc is None:
        raw_d = os.environ.get("ISKRA_DOCUMENTS_PARQUET", "").strip()
        pq_doc = Path(raw_d) if raw_d else _ROOT / "out" / "documents.parquet"
    pq_doc = pq_doc.resolve()

    pq_chunk = args.chunks_parquet
    if pq_chunk is None:
        raw_c = os.environ.get("ISKRA_CHUNKS_PARQUET", "").strip()
        pq_chunk = Path(raw_c) if raw_c else _ROOT / "out" / "chunks.parquet"
    pq_chunk = pq_chunk.resolve()

    id_start = default_document_id_start()
    id_by_rel = document_id_by_rel_path(corpus, glob_pattern=args.glob_pattern, id_start=id_start)

    if not args.quiet:
        print(f"document_id 映射: {len(id_by_rel)} 篇（起始于 {id_start}）", flush=True)

    rows = build_documents_rows(corpus, id_by_rel)
    write_documents_parquet(rows, pq_doc)
    if not args.quiet:
        print(f"已写 {pq_doc}", flush=True)

    expect_dim: int | None = None
    if not args.skip_validate_dim:
        dim_raw = os.environ.get("ISKRA_EMBED_DIM", "1024").strip()
        expect_dim = int(dim_raw)

    try:
        n, dim = write_chunks_parquet_from_jsonl_and_npy_and_txt(
            chunks_jsonl=jsl,
            embeddings_npy=npy,
            tokenized_txt=txt,
            document_id_by_rel=id_by_rel,
            path=pq_chunk,
            expect_dim=expect_dim,
            max_chunks=args.max_chunks,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)

    if not args.quiet:
        print(f"已写 {pq_chunk}  块数={n}  dim={dim}", flush=True)


if __name__ == "__main__":
    main()
