"""CLI：读 ``chunks.jsonl`` → Sentence-Transformers 批编码 → 行对齐写出 ``.npy``。

与 JSONL **第 i 行**对应 **``embeddings[i]``**；`export_parquet` 依此组装 ``chunks.parquet``。

``ISKRA_EMBED_BATCH_SIZE``（默认 48）；CUDA OOM 时自动缩小窗口 batch（如 48→32），避免整次任务失败。。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")


def main() -> None:

    import os

    from iskra_etl.embedder import (
        embed_chunk_records,
        iter_chunk_records_from_jsonl,
        load_sentence_model,
        save_chunk_embeddings_npy,
        validate_embedding_dim,
    )

    ap = argparse.ArgumentParser(description="chunks.jsonl → 行对齐 chunks_embeddings.npy")
    ap.add_argument(
        "--input",
        type=Path,
        default=None,
        help="chunks.jsonl（默认 env ISKRA_CHUNK_JSONL 或 out/chunks.jsonl）",
    )
    ap.add_argument(
        "--output-npy",
        type=Path,
        default=None,
        help="输出 .npy（默认 ISKRA_CHUNK_EMBEDDINGS_NPY 或 out/chunks_embeddings.npy）",
    )
    ap.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="只编码前 N 条（与 export 时 --max-chunks 须一致）",
    )
    ap.add_argument(
        "--validate-dim",
        action="store_true",
        help="与 ISKRA_EMBED_DIM（默认 1024）核对模型输出维数",
    )
    ap.add_argument(
        "--no-save-npy",
        action="store_true",
        help="只跑编码与统计，不写 .npy（冒烟）",
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

    out_npy = args.output_npy
    if out_npy is None:
        raw_o = os.environ.get("ISKRA_CHUNK_EMBEDDINGS_NPY", "").strip()
        out_npy = Path(raw_o) if raw_o else _ROOT / "out" / "chunks_embeddings.npy"
    out_npy = out_npy.resolve()

    model = load_sentence_model()
    if args.validate_dim:
        dim_raw = os.environ.get("ISKRA_EMBED_DIM", "1024").strip()
        validate_embedding_dim(model, expected=int(dim_raw))

    records = []
    for i, r in enumerate(iter_chunk_records_from_jsonl(src)):
        if args.max_chunks is not None and i >= args.max_chunks:
            break
        records.append(r)

    if not records:
        print("0 条块，未编码。", file=sys.stderr)
        sys.exit(0)

    if not args.quiet:
        print(
            f"编码 {len(records)} 条  device={model.device!s}  dim={model.get_embedding_dimension()}",
            flush=True,
        )

    embs = embed_chunk_records(records, model, show_progress_bar=not args.quiet)
    norms = np.linalg.norm(embs, axis=1)
    if not args.quiet:
        print(
            f"OK  shape={tuple(embs.shape)}  L2(mean)={float(norms.mean()):.6f}  L2(min)={float(norms.min()):.6f}",
            flush=True,
        )

    if not args.no_save_npy:
        save_chunk_embeddings_npy(out_npy, embs)
        if not args.quiet:
            print(f"已写 {out_npy}", flush=True)


if __name__ == "__main__":
    main()
