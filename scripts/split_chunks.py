"""CLI：在语料根目录下按 glob 找多篇 ``index.md``（每篇即一个文档），切块写 JSONL（不向量化）。

「语料根」= 存放整棵文档树的总目录；「一篇文档」= 匹配到的一个 ``index.md`` 文件
（不是指数据库里的 document 表名，只是日常说法与 LlamaIndex 的 Document 一致）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from iskra_etl.splitter import glob_index_paths, split_corpus_to_chunks, write_chunk_jsonl

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")


def main() -> None:

    ap = argparse.ArgumentParser(
        description="在语料根下按 glob 扫描多篇 index.md，切块输出 JSONL（无 Embedding）",
    )
    ap.add_argument(
        "--corpus-root",
        "--documents-root",
        type=Path,
        dest="corpus_root",
        default=None,
        help="语料根目录：下面有多篇 **/index.md（默认 env ISKRA_CORPUS_ROOT，再默认同级 iskra-data）",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 chunks.jsonl（默认 env ISKRA_CHUNK_JSONL）",
    )
    ap.add_argument(
        "--header-path-separator",
        type=str,
        default=None,
        help="Markdown 标题路径分隔符（默认 env ISKRA_HEADER_PATH_SEP 或 /）",
    )
    ap.add_argument(
        "--glob",
        dest="glob_pattern",
        default="**/index.md",
        help="相对语料根的 glob，每命中一个文件即一篇文档",
    )
    ap.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="不打印进度，只打印最终一行摘要",
    )
    args = ap.parse_args()

    import os

    corpus_root = args.corpus_root
    if corpus_root is None:
        raw = os.environ.get("ISKRA_CORPUS_ROOT", "").strip()
        corpus_root = Path(raw) if raw else _ROOT.parent / "iskra-data"
    corpus_root = corpus_root.resolve()

    if not corpus_root.is_dir():
        print(f"语料根不存在或不是目录: {corpus_root}", file=sys.stderr)
        sys.exit(2)

    out = args.output
    if out is None:
        raw_o = os.environ.get("ISKRA_CHUNK_JSONL", "").strip()
        out = Path(raw_o) if raw_o else _ROOT / "out" / "chunks.jsonl"

    header_sep = (
        args.header_path_separator
        if args.header_path_separator is not None
        else os.environ.get("ISKRA_HEADER_PATH_SEP", "/")
    )

    quiet = args.quiet
    paths = glob_index_paths(corpus_root, args.glob_pattern)
    if not quiet:
        print(f"语料根（文档树目录）: {corpus_root}", flush=True)
        print(f'glob "{args.glob_pattern}" → 文档篇数（每个 index.md 一篇）: {len(paths)}', flush=True)

    if not paths:
        if not quiet:
            print("没有匹配的文档，未写入文件。", flush=True)
        sys.exit(0)

    def _on_document_start(total: int, idx_1: int, rel_path: str) -> None:
        if quiet:
            return
        print(f"  [{idx_1}/{total}] 正在按 Markdown 标题切段  {rel_path}", flush=True)

    def _on_document_done(total: int, idx_1: int, rel_path: str, chunks: int) -> None:
        if quiet:
            return
        print(f"  [{idx_1}/{total}] 本篇完成  →  {chunks} 块", flush=True)

    if not quiet:
        print(f"载入 {len(paths)} 篇文档到内存（LlamaIndex）…", flush=True)

    records_iter = split_corpus_to_chunks(
        corpus_root,
        glob_pattern=args.glob_pattern,
        paths=paths,
        header_path_separator=header_sep,
        on_document_start=_on_document_start,
        on_document_done=_on_document_done,
    )

    if not quiet:
        print(
            f"已开始按 Markdown 标题切段并写入 JSONL（每本篇内 chunk_index 从 0）→ {out}",
            flush=True,
        )

    n = write_chunk_jsonl(out, records_iter)

    print(f"完成  语料根={corpus_root}  JSONL行数={n}（块总数）→ {out}", flush=True)


if __name__ == "__main__":
    main()
