"""一次性/运维脚本：回填 ``chunk.tokenized_fts``。
虽然是主流程之外，但是可以重复执行，仅仅更新 tokenized_fts 这一列。

默认全流程：

1. ``prepare``：读 ``chunks.parquet`` 的 ``document_id`` / ``chunk_index``，
   与 ``chunks_tokenized.txt`` 按行合并为 ``chunks_tokenized.tsv``。
2. ``check``：检查 TSV 行数、重复键、空分词行，并打印前几行样例。
3. ``apply``：经 SSH 隧道连接 PostgreSQL，把 TSV COPY 到临时 staging 表，
   再用 ``to_tsvector('simple', tokenized_text)`` 更新 ``chunk.tokenized_fts``

也可只执行一个阶段：

``uv run python scripts/backfill_chunk_text_search_vector.py prepare``
``uv run python scripts/backfill_chunk_text_search_vector.py check``
``uv run python scripts/backfill_chunk_text_search_vector.py apply``
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import psycopg
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")

BackfillMode = Literal["prepare", "check", "apply", "all"]

TSV_COLUMNS = ("document_id", "chunk_index", "tokenized_text")


@dataclass(frozen=True)
class BackfillPaths:
    chunks_parquet: Path
    tokenized_txt: Path
    tokenized_tsv: Path


@dataclass(frozen=True)
class TsvStats:
    rows: int
    empty_tokenized: int
    duplicated_keys: int
    min_document_id: int | None
    max_document_id: int | None
    sample_rows: list[tuple[int, int, str]]


def _default_paths(args: argparse.Namespace) -> BackfillPaths:
    chunks_parquet = args.chunks_parquet
    if chunks_parquet is None:
        raw = os.environ.get("ISKRA_CHUNKS_PARQUET", "").strip()
        chunks_parquet = Path(raw) if raw else _ROOT / "out" / "chunks.parquet"

    tokenized_txt = args.tokenized_txt
    if tokenized_txt is None:
        raw = os.environ.get("ISKRA_CHUNK_TOKENIZED_TXT", "").strip()
        tokenized_txt = Path(raw) if raw else _ROOT / "out" / "chunks_tokenized.txt"

    tokenized_tsv = args.tokenized_tsv
    if tokenized_tsv is None:
        raw = os.environ.get("ISKRA_CHUNK_TOKENIZED_TSV", "").strip()
        tokenized_tsv = Path(raw) if raw else _ROOT / "out" / "chunks_tokenized.tsv"

    return BackfillPaths(
        chunks_parquet=chunks_parquet.resolve(),
        tokenized_txt=tokenized_txt.resolve(),
        tokenized_tsv=tokenized_tsv.resolve(),
    )


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        msg = f"{label} 不存在: {path}"
        raise FileNotFoundError(msg)


def _read_tokenized_texts(path: Path, *, max_chunks: int | None = None) -> list[str]:
    lines: list[str] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_chunks is not None and i >= max_chunks:
                break
            lines.append(line.rstrip("\n\r"))
    return lines


def _read_chunk_keys(path: Path, *, max_chunks: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["document_id", "chunk_index"])
    if max_chunks is not None:
        df = df.iloc[:max_chunks]
    missing = [c for c in ("document_id", "chunk_index") if c not in df.columns]
    if missing:
        msg = f"chunks.parquet 缺少列: {missing}"
        raise ValueError(msg)
    return df


def _iter_tsv_rows(path: Path) -> Iterable[tuple[int, int, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != list(TSV_COLUMNS):
            msg = f"TSV 表头应为 {list(TSV_COLUMNS)}，实际为 {reader.fieldnames}"
            raise ValueError(msg)
        for row in reader:
            yield (
                int(row["document_id"]),
                int(row["chunk_index"]),
                row["tokenized_text"],
            )


def prepare(paths: BackfillPaths, *, max_chunks: int | None = None) -> int:
    """生成 ``chunks_tokenized.tsv``，但不连接数据库。"""
    _require_file(paths.chunks_parquet, "chunks.parquet")
    _require_file(paths.tokenized_txt, "chunks_tokenized.txt")

    print(f"读取 chunk 键: {paths.chunks_parquet}", flush=True)
    keys = _read_chunk_keys(paths.chunks_parquet, max_chunks=max_chunks)
    print(f"读取分词文本: {paths.tokenized_txt}", flush=True)
    tokenized_texts = _read_tokenized_texts(paths.tokenized_txt, max_chunks=max_chunks)

    if len(keys) != len(tokenized_texts):
        msg = (
            f"行数不一致：chunks.parquet 键行数={len(keys)}，"
            f"chunks_tokenized.txt 行数={len(tokenized_texts)}"
        )
        raise ValueError(msg)

    duplicated = int(keys.duplicated(subset=["document_id", "chunk_index"]).sum())
    if duplicated:
        msg = f"chunks.parquet 中 document_id/chunk_index 重复: {duplicated}"
        raise ValueError(msg)

    paths.tokenized_tsv.parent.mkdir(parents=True, exist_ok=True)
    with paths.tokenized_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(TSV_COLUMNS)
        for row, tokenized_text in zip(keys.itertuples(index=False), tokenized_texts, strict=True):
            writer.writerow((int(row.document_id), int(row.chunk_index), tokenized_text))

    print(f"已写 TSV: {paths.tokenized_tsv}  行数={len(keys)}", flush=True)
    print("可先执行 check 检查该 TSV；默认全流程会继续检查并回填。", flush=True)
    return int(len(keys))


def _analyze_tsv(paths: BackfillPaths, *, sample: int = 0) -> TsvStats:
    """扫描 TSV、校验（空表/重复键）；不打印。``sample > 0`` 时收集样例行。"""
    _require_file(paths.tokenized_tsv, "chunks_tokenized.tsv")

    keys: set[tuple[int, int]] = set()
    duplicated = 0
    empty = 0
    rows = 0
    min_doc: int | None = None
    max_doc: int | None = None
    samples: list[tuple[int, int, str]] = []

    for document_id, chunk_index, tokenized in _iter_tsv_rows(paths.tokenized_tsv):
        key = (document_id, chunk_index)
        if key in keys:
            duplicated += 1
        else:
            keys.add(key)
        if tokenized == "":
            empty += 1
        min_doc = document_id if min_doc is None else min(min_doc, document_id)
        max_doc = document_id if max_doc is None else max(max_doc, document_id)
        rows += 1
        if sample > 0 and len(samples) < sample:
            samples.append((document_id, chunk_index, tokenized))

    if rows == 0:
        msg = "TSV 无数据，拒绝 apply"
        raise ValueError(msg)
    if duplicated:
        msg = f"TSV 存在重复键，拒绝 apply: {duplicated}"
        raise ValueError(msg)

    return TsvStats(
        rows=rows,
        empty_tokenized=empty,
        duplicated_keys=duplicated,
        min_document_id=min_doc,
        max_document_id=max_doc,
        sample_rows=samples,
    )


def check(paths: BackfillPaths, *, sample: int = 5) -> TsvStats:
    """检查 ``chunks_tokenized.tsv`` 并校验；打印便于人工确认的摘要。"""
    stats = _analyze_tsv(paths, sample=sample)

    print(f"TSV: {paths.tokenized_tsv}", flush=True)
    print(f"行数: {stats.rows}", flush=True)
    print(f"空 tokenized_text 行数: {stats.empty_tokenized}", flush=True)
    print(f"重复 (document_id, chunk_index) 数: {stats.duplicated_keys}", flush=True)
    print(f"document_id 范围: {stats.min_document_id}..{stats.max_document_id}", flush=True)
    print("样例:", flush=True)
    for document_id, chunk_index, tokenized in stats.sample_rows:
        preview = tokenized[:160]
        print(f"  document_id={document_id} chunk_index={chunk_index} tokenized={preview}", flush=True)

    return stats


def _copy_tsv_to_staging(cur: psycopg.Cursor, paths: BackfillPaths) -> int:
    cur.execute(
        """
        CREATE TEMP TABLE chunk_text_search_vector_backfill (
            document_id bigint NOT NULL,
            chunk_index integer NOT NULL,
            tokenized_text text NOT NULL
        ) ON COMMIT DROP
        """
    )

    n = 0
    with cur.copy(
        """
        COPY chunk_text_search_vector_backfill (document_id, chunk_index, tokenized_text)
        FROM STDIN
        """
    ) as copy:
        for document_id, chunk_index, tokenized_text in _iter_tsv_rows(paths.tokenized_tsv):
            copy.write_row((document_id, chunk_index, tokenized_text))
            n += 1
    return n


def apply(paths: BackfillPaths) -> tuple[int, int, int]:
    """把 TSV COPY 到临时 staging 表，并回填 ``chunk.tokenized_fts``。"""
    from iskra_etl.loader import (
        build_conninfo_localhost,
        open_ssh_tunnel,
        pg_conn_config_from_env,
        ssh_tunnel_config_from_env,
    )

    stats = _analyze_tsv(paths)

    tunnel_cfg = ssh_tunnel_config_from_env()
    pg_cfg = pg_conn_config_from_env()
    tunnel = open_ssh_tunnel(tunnel_cfg)
    tunnel.start()
    try:
        local_port = int(tunnel.local_bind_port)
        conninfo = build_conninfo_localhost(local_port, pg_cfg)
        print(f"SSH 隧道 127.0.0.1:{local_port} → {tunnel_cfg.remote_pg_host}:{tunnel_cfg.remote_pg_port}", flush=True)
        with psycopg.connect(conninfo) as conn:
            conn.execute("SET timezone TO 'UTC'")
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM chunk")
                chunk_count = int(cur.fetchone()[0])
                print(f"数据库 chunk 行数: {chunk_count}", flush=True)
                if chunk_count != stats.rows:
                    msg = f"数据库 chunk 行数 {chunk_count} 与 TSV 行数 {stats.rows} 不一致，拒绝回填"
                    raise ValueError(msg)

                print(f"COPY TSV → TEMP chunk_text_search_vector_backfill …", flush=True)
                staged = _copy_tsv_to_staging(cur, paths)
                cur.execute(f"SELECT count(*) FROM chunk_text_search_vector_backfill")
                staged_count = int(cur.fetchone()[0])
                if staged != staged_count or staged_count != stats.rows:
                    msg = f"staging 行数异常：写入={staged}, 表内={staged_count}, TSV={stats.rows}"
                    raise ValueError(msg)

                print("UPDATE chunk.tokenized_fts …", flush=True)
                cur.execute(
                    f"""
                    UPDATE chunk AS c
                    SET tokenized_fts = to_tsvector('simple', temp.tokenized_text)
                    FROM chunk_text_search_vector_backfill AS temp
                    WHERE c.document_id = temp.document_id
                      AND c.chunk_index = temp.chunk_index
                    """
                )
                updated = int(cur.rowcount)
                if updated != stats.rows:
                    msg = f"UPDATE 命中行数 {updated} 与 TSV 行数 {stats.rows} 不一致，回滚"
                    raise ValueError(msg)

                cur.execute("SELECT count(*) FROM chunk WHERE tokenized_fts IS NULL")
                nulls = int(cur.fetchone()[0])
                print(f"UPDATE 命中: {updated}；tokenized_fts IS NULL: {nulls}", flush=True)
            conn.commit()
            print("已提交。TEMP staging 表将在提交后自动删除。", flush=True)
            return stats.rows, updated, nulls
    finally:
        tunnel.stop()


def run_all(paths: BackfillPaths, *, max_chunks: int | None = None, sample: int = 5) -> None:
    prepare(paths, max_chunks=max_chunks)
    check(paths, sample=sample)
    apply(paths)


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="prepare/check/apply 回填 chunk.tokenized_fts（默认全流程）",
    )
    ap.add_argument(
        "mode",
        nargs="?",
        choices=["prepare", "check", "apply", "all"],
        default="all",
        help="默认 all；也可只执行 prepare / check / apply",
    )
    ap.add_argument(
        "--chunks-parquet",
        type=Path,
        default=None,
        help="chunks.parquet（默认 ISKRA_CHUNKS_PARQUET 或 out/chunks.parquet）",
    )
    ap.add_argument(
        "--tokenized-txt",
        type=Path,
        default=None,
        help="chunks_tokenized.txt（默认 ISKRA_CHUNK_TOKENIZED_TXT 或 out/chunks_tokenized.txt）",
    )
    ap.add_argument(
        "--tokenized-tsv",
        type=Path,
        default=None,
        help="输出/读取 TSV（默认 ISKRA_CHUNK_TOKENIZED_TSV 或 out/chunks_tokenized.tsv）",
    )
    ap.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="仅 prepare 前 N 行；调试用，apply 前数据库行数也必须匹配",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=5,
        help="check 打印样例行数",
    )
    return ap


def main() -> None:
    args = _build_arg_parser().parse_args()
    paths = _default_paths(args)

    try:
        mode: BackfillMode = args.mode
        if mode == "prepare":
            prepare(paths, max_chunks=args.max_chunks)
        elif mode == "check":
            check(paths, sample=args.sample)
        elif mode == "apply":
            apply(paths)
        else:
            run_all(paths, max_chunks=args.max_chunks, sample=args.sample)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, psycopg.Error) as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

