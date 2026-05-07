"""笔记本经 SSH 隧道 ``COPY`` 灌入 ``documents.parquet`` + ``chunks.parquet``。

环境变量见 ``.env.example``（``ISKRA_SSH_*``、``ISKRA_PG_*``）。
非空库须 ``--truncate`` 或 ``ISKRA_LOAD_TRUNCATE=1``。
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
    from iskra_etl.loader import (
        load_parquets_via_tunnel_from_env,
        pg_conn_config_from_env,
        ssh_tunnel_config_from_env,
    )

    ap = argparse.ArgumentParser(description="SSH 隧道 + COPY 加载 Parquet 至远程 PostgreSQL")
    ap.add_argument(
        "--documents-parquet",
        type=Path,
        default=None,
        help="documents.parquet（默认 ISKRA_DOCUMENTS_PARQUET 或 out/documents.parquet）",
    )
    ap.add_argument(
        "--chunks-parquet",
        type=Path,
        default=None,
        help="chunks.parquet（默认 ISKRA_CHUNKS_PARQUET 或 out/chunks.parquet）",
    )
    ap.add_argument(
        "--truncate",
        action="store_true",
        help="非空库时允许 TRUNCATE document CASCADE 后全量导入",
    )
    ap.add_argument(
        "--embed-dim",
        type=int,
        default=None,
        help="向量维数（默认 ISKRA_EMBED_DIM 或 1024）",
    )
    ap.add_argument(
        "--dry-run-config",
        action="store_true",
        help="只解析环境并打印将使用的路径，不连 SSH、不写库",
    )
    args = ap.parse_args()

    if args.dry_run_config:
        tc = ssh_tunnel_config_from_env()
        pg = pg_conn_config_from_env()
        dp = args.documents_parquet or Path(
            os.environ.get("ISKRA_DOCUMENTS_PARQUET", "").strip() or (_ROOT / "out" / "documents.parquet"),
        )
        cp = args.chunks_parquet or Path(
            os.environ.get("ISKRA_CHUNKS_PARQUET", "").strip() or (_ROOT / "out" / "chunks.parquet"),
        )
        print(f"SSH 跳板: {tc.user}@{tc.jump_host}:{tc.jump_port} → {tc.remote_pg_host}:{tc.remote_pg_port}")
        print(f"PostgreSQL: user={pg.user} db={pg.dbname} sslmode={pg.sslmode}")
        print(f"documents: {dp.resolve()}")
        print(f"chunks:    {cp.resolve()}")
        print(f"truncate:  {args.truncate}")
        return

    try:
        load_parquets_via_tunnel_from_env(
            documents_parquet=args.documents_parquet,
            chunks_parquet=args.chunks_parquet,
            truncate=args.truncate if args.truncate else None,
            expect_embedding_dim=args.embed_dim,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
