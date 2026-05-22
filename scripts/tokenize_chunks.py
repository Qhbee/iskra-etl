"""CLI：读 ``chunks.jsonl`` → jieba 分词 + 停用词 → 行对齐 ``chunks_tokenized.txt``。

与 JSONL **第 i 行**、``chunks_embeddings.npy`` 的 **``embeddings[i]``** 严格同序；
后续 ``export_parquet`` 再合并 jsonl + txt + npy（本脚本不写 Parquet）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")


def main() -> None:
    import os

    from iskra_etl.tokenizer import tokenize_jsonl_to_txt

    ap = argparse.ArgumentParser(
        description="chunks.jsonl → 行对齐 chunks_tokenized.txt（jieba + 停用词）",
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=None,
        help="chunks.jsonl（默认 env ISKRA_CHUNK_JSONL 或 out/chunks.jsonl）",
    )
    ap.add_argument(
        "--output-txt",
        type=Path,
        default=None,
        help="输出 txt（默认 ISKRA_CHUNK_TOKENIZED_TXT 或 out/chunks_tokenized.txt）",
    )
    ap.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="只处理前 N 条（与 embed / export 时 --max-chunks 须一致）",
    )
    ap.add_argument(
        "--quiet",
        "-q",
        action="store_true",
    )
    args = ap.parse_args()

    src = args.input
    if src is None:
        raw = os.environ.get("ISKRA_CHUNK_JSONL", "").strip()
        src = Path(raw) if raw else _ROOT / "out" / "chunks.jsonl"

    src = src.resolve()
    if not src.is_file():
        print(f"输入文件不存在: {src}", file=sys.stderr)
        sys.exit(2)

    out_txt = args.output_txt
    if out_txt is None:
        raw_o = os.environ.get("ISKRA_CHUNK_TOKENIZED_TXT", "").strip()
        out_txt = Path(raw_o) if raw_o else _ROOT / "out" / "chunks_tokenized.txt"
    out_txt = out_txt.resolve()

    if not args.quiet:
        print(f"分词  输入={src}  输出={out_txt}", flush=True)

    n = tokenize_jsonl_to_txt(
        src,
        out_txt,
        max_chunks=args.max_chunks,
        show_progress_bar=not args.quiet,
    )

    if n == 0:
        print("0 条块，未写入。", file=sys.stderr)
        sys.exit(0)

    if not args.quiet:
        print(f"OK  行数={n}  → {out_txt}", flush=True)


if __name__ == "__main__":
    main()
