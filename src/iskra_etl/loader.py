"""笔记本直连远程 PostgreSQL：SSH 隧道 + psycopg ``COPY`` 灌入 Parquet。

与 :mod:`iskra_etl.exporter` 产出的两表对齐：``document``、``chunk``（``001_init.sql``）。
需在目标库已安装 ``pgvector`` 且表已存在。

非空库全量重载须显式 ``truncate=True``（或 CLI ``--truncate``），将
``TRUNCATE document RESTART IDENTITY CASCADE`` 并写入新数据，然后 ``setval`` 对齐序列。
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
from pgvector.psycopg import register_vector
from sshtunnel import SSHTunnelForwarder


DOCUMENT_COPY_BATCH_SIZE = 5_000
CHUNK_COPY_BATCH_SIZE = 10_000

@dataclass(frozen=True)
class SshTunnelConfig:
    jump_host: str
    jump_port: int = 22
    user: str = ""
    pkey_path: Path | None = None
    password: str | None = None
    remote_pg_host: str = "127.0.0.1"
    remote_pg_port: int = 5432
    local_bind_port: int = 0
    set_keepalive: float = 30.0


@dataclass(frozen=True)
class PgConnConfig:
    user: str
    password: str
    dbname: str
    sslmode: str = "prefer"


def ssh_tunnel_config_from_env() -> SshTunnelConfig:
    host = os.environ.get("ISKRA_SSH_JUMP_HOST", "").strip()
    if not host:
        msg = "请设置 ISKRA_SSH_JUMP_HOST（跳板机）"
        raise OSError(msg)
    port = int(os.environ.get("ISKRA_SSH_JUMP_PORT", "22"))
    user = os.environ.get("ISKRA_SSH_USER", "").strip()
    if not user:
        msg = "请设置 ISKRA_SSH_USER"
        raise OSError(msg)
    key_raw = os.environ.get("ISKRA_SSH_KEY_PATH", "").strip()
    pkey = Path(os.path.expandvars(key_raw)).expanduser() if key_raw else None
    if pkey is not None and not pkey.is_file():
        msg = f"SSH 私钥不存在: {pkey}"
        raise OSError(msg)
    pwd = os.environ.get("ISKRA_SSH_PASSWORD", "").strip() or None
    r_host = os.environ.get("ISKRA_PG_REMOTE_HOST", "127.0.0.1").strip()
    r_port = int(os.environ.get("ISKRA_PG_REMOTE_PORT", "5432"))
    loc = os.environ.get("ISKRA_PG_LOCAL_BIND_PORT", "").strip()
    local_port = int(loc) if loc else 0
    return SshTunnelConfig(
        jump_host=host,
        jump_port=port,
        user=user,
        pkey_path=pkey,
        password=pwd,
        remote_pg_host=r_host,
        remote_pg_port=r_port,
        local_bind_port=local_port,
        set_keepalive=float(os.environ.get("ISKRA_SSH_KEEPALIVE", "30")),
    )


def pg_conn_config_from_env() -> PgConnConfig:
    user = os.environ.get("ISKRA_PG_USER", "").strip()
    db = os.environ.get("ISKRA_PG_DB", "").strip()
    pwd = os.environ.get("ISKRA_PG_PASSWORD", "").strip()
    if not user or not db:
        msg = "请设置 ISKRA_PG_USER、ISKRA_PG_DB（及 ISKRA_PG_PASSWORD）"
        raise OSError(msg)
    sslmode = os.environ.get("ISKRA_PG_SSLMODE", "prefer").strip()
    return PgConnConfig(user=user, password=pwd, dbname=db, sslmode=sslmode)


def build_conninfo_localhost(port: int, cfg: PgConnConfig) -> str:
    return psycopg.conninfo.make_conninfo(
        host="127.0.0.1",
        port=port,
        user=cfg.user,
        password=cfg.password or None,
        dbname=cfg.dbname,
        sslmode=cfg.sslmode,
    )


def open_ssh_tunnel(tcfg: SshTunnelConfig) -> SSHTunnelForwarder:
    """启动本地 → 跳板 → ``remote_pg_host:remote_pg_port`` 的转发。"""
    kwargs: dict[str, Any] = {
        "ssh_address_or_host": (tcfg.jump_host, tcfg.jump_port),
        "ssh_username": tcfg.user,
        "remote_bind_address": (tcfg.remote_pg_host, tcfg.remote_pg_port),
        "local_bind_address": ("127.0.0.1", tcfg.local_bind_port),
        "set_keepalive": int(max(tcfg.set_keepalive, 5)),
    }
    if tcfg.pkey_path is not None:
        kwargs["ssh_pkey"] = str(tcfg.pkey_path)
    if tcfg.password is not None:
        kwargs["ssh_password"] = tcfg.password
    return SSHTunnelForwarder(**kwargs)


def _db_nonempty(cur: psycopg.Cursor) -> bool:
    cur.execute("SELECT EXISTS (SELECT 1 FROM document LIMIT 1)")
    d = cur.fetchone()
    cur.execute("SELECT EXISTS (SELECT 1 FROM chunk LIMIT 1)")
    c = cur.fetchone()
    return bool(d and d[0]) or bool(c and c[0])


def _truncate_tables(cur: psycopg.Cursor) -> None:
    """清空 document/chunk 并重置两表 ``bigserial`` 序列（下一条隐式 id 从 1 起）。"""
    cur.execute("TRUNCATE document RESTART IDENTITY CASCADE")
    # RESTART IDENTITY 已对参与 TRUNCATE 的表都重置序列；不确定的话可以再 setval 一次，避免极少数清空下序列未对齐（一般不需要）
    # cur.execute("SELECT setval(pg_get_serial_sequence('document', 'id'), 1, false)")
    # cur.execute("SELECT setval(pg_get_serial_sequence('chunk', 'id'), 1, false)")


def _setval_sequences(cur: psycopg.Cursor) -> None:
    cur.execute("SELECT COALESCE(MAX(id), 1) FROM document")
    max_doc = cur.fetchone()[0]
    cur.execute("SELECT COALESCE(MAX(id), 1) FROM chunk")
    max_chunk = cur.fetchone()[0]
    cur.execute(
        "SELECT setval(pg_get_serial_sequence('document', 'id'), %s, true)",
        (int(max_doc),),
    )
    cur.execute(
        "SELECT setval(pg_get_serial_sequence('chunk', 'id'), %s, true)",
        (int(max_chunk),),
    )


def _iter_copy_rows(df: pd.DataFrame, *, desc: str) -> Iterator[Any]:
    """``df.itertuples(index=False)`` 外包 ``tqdm``；无 tqdm 包则裸迭代。"""
    total = len(df)
    base = df.itertuples(index=False)
    if total == 0:
        yield from ()
        return
    try:
        from tqdm.auto import tqdm

        yield from tqdm(
            base,
            total=total,
            desc=desc,
            unit="行",
            mininterval=0.2,
            dynamic_ncols=True,
        )
    except ImportError:
        yield from base


def _nullable_str(v: Any) -> str | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except TypeError:
        pass
    if isinstance(v, float) and np.isnan(v):
        return None
    s = str(v)
    return s if s else None


def _copy_documents(cur: psycopg.Cursor, df: pd.DataFrame) -> int:
    cols = ["id", "rel_path", "title", "book", "full_text", "content_sha256"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        msg = f"documents.parquet 缺少列: {missing}"
        raise ValueError(msg)
    n = 0
    sql = """COPY document (id, rel_path, title, book, full_text, content_sha256) FROM STDIN"""
    with cur.copy(sql) as copy:
        for row in _iter_copy_rows(df, desc="COPY document"):
            copy.write_row(
                (
                    int(row.id),
                    str(row.rel_path),
                    _nullable_str(row.title),
                    _nullable_str(row.book),
                    str(row.full_text),
                    str(row.content_sha256),
                )
            )
            n += 1
        print(
            "  … 本地已向 psycopg 的 COPY 流写完所有行；进度条只统计发送速度。正在等待服务器 PostgreSQL 结束 COPY 并落库（经 SSH 隧道刷盘，可能还需一会儿）…",
            flush=True,
        )
    print("  document 本批次 COPY 已在服务器侧完成。", flush=True)
    return n


def _embedding_to_list(v: Any, *, expect_dim: int) -> list[float]:
    if hasattr(v, "tolist"):
        v = v.tolist()
    elif isinstance(v, np.ndarray):
        v = v.tolist()
    if not isinstance(v, (list, tuple)):
        msg = f"embedding 不是列表: {type(v)}"
        raise TypeError(msg)
    if len(v) != expect_dim:
        msg = f"embedding 维数 {len(v)} ≠ {expect_dim}"
        raise ValueError(msg)
    return [float(x) for x in v]


def _embedding_to_pgvector_copy_literal(emb: list[float]) -> str:
    """pgvector 的文本输入须以 ``[`` 开头；psycopg ``COPY`` 会把 Python ``list`` 编成 ``{a,b,c}`` 数组字面量，与 ``vector`` 类型不兼容。"""
    return json.dumps(emb, separators=(",", ":"))


def _copy_chunks(cur: psycopg.Cursor, df: pd.DataFrame, *, expect_dim: int) -> int:
    need = ["document_id", "chunk_index", "text", "embedding"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        msg = f"chunks.parquet 缺少列: {missing}"
        raise ValueError(msg)
    n = 0
    sql = """COPY chunk (document_id, chunk_index, text, embedding) FROM STDIN"""
    with cur.copy(sql) as copy:
        for row in _iter_copy_rows(df, desc="COPY chunk"):
            emb = _embedding_to_list(row.embedding, expect_dim=expect_dim)
            lit = _embedding_to_pgvector_copy_literal(emb)
            copy.write_row((int(row.document_id), int(row.chunk_index), str(row.text), lit))
            n += 1
        print(
            "  … 本地已向 psycopg 的 COPY 流写完所有行；进度条只统计发送速度。正在等待服务器 PostgreSQL 结束 COPY 并落库（经 SSH 隧道刷盘，可能还需一会儿）…",
            flush=True,
        )
    print("  chunk 本批次 COPY 已在服务器侧完成。", flush=True)
    return n


def _copy_dataframe_in_batches(
    conn: psycopg.Connection,
    df: pd.DataFrame,
    *,
    label: str,
    batch_size: int,
    copy_fn: Callable[[psycopg.Cursor, pd.DataFrame], int],
) -> int:
    if batch_size <= 0:
        msg = f"{label} COPY 批大小必须为正数: {batch_size}"
        raise ValueError(msg)

    total = len(df)
    copied = 0
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        print(f"COPY {label} 批次 {start + 1}-{end}/{total} …", flush=True)
        batch = df.iloc[start:end]
        with conn.cursor() as cur:
            copied += copy_fn(cur, batch)
        conn.commit()
        print(f"  {label} 批次已提交：{copied}/{total}", flush=True)
    return copied


def load_parquets_via_tunnel(
    documents_parquet: Path | str,
    chunks_parquet: Path | str,
    *,
    tunnel_cfg: SshTunnelConfig,
    pg_cfg: PgConnConfig,
    truncate: bool = False,
    expect_embedding_dim: int | None = None,
) -> tuple[int, int]:
    """经 SSH 隧道打开连接，``COPY`` 两个 Parquet；``COPY`` 循环带 ``tqdm`` 进度条；返回 ``(document 行数, chunk 行数)``。"""
    exp_dim = expect_embedding_dim
    if exp_dim is None:
        raw = os.environ.get("ISKRA_EMBED_DIM", "1024").strip()
        exp_dim = int(raw) if raw else 1024

    doc_path = Path(documents_parquet)
    chk_path = Path(chunks_parquet)
    if not doc_path.is_file():
        msg = f"不存在: {doc_path}"
        raise FileNotFoundError(msg)
    if not chk_path.is_file():
        msg = f"不存在: {chk_path}"
        raise FileNotFoundError(msg)

    print(f"读取 Parquet: {doc_path}", flush=True)
    df_doc = pd.read_parquet(doc_path)
    print(f"读取 Parquet: {chk_path}", flush=True)
    df_chunk = pd.read_parquet(chk_path)

    tunnel = open_ssh_tunnel(tunnel_cfg)
    tunnel.start()
    try:
        local_port = int(tunnel.local_bind_port)
        conninfo = build_conninfo_localhost(local_port, pg_cfg)
        print(f"SSH 隧道 127.0.0.1:{local_port} → {tunnel_cfg.remote_pg_host}:{tunnel_cfg.remote_pg_port}", flush=True)
        with psycopg.connect(conninfo) as conn:
            register_vector(conn)
            conn.execute("SET timezone TO 'UTC'")
            with conn.cursor() as cur:
                if _db_nonempty(cur):
                    if not truncate:
                        msg = (
                            "目标库 document/chunk 已有数据。全量重导请传 truncate=True "
                            "或设置 ISKRA_LOAD_TRUNCATE=1，并执行 TRUNCATE document CASCADE。"
                        )
                        raise RuntimeError(msg)
                    print("TRUNCATE document RESTART IDENTITY CASCADE …", flush=True)
                    _truncate_tables(cur)
                    conn.commit()
                    print("TRUNCATE 已提交。", flush=True)

            print(f"COPY document（{len(df_doc)} 行, 每批 {DOCUMENT_COPY_BATCH_SIZE}）…", flush=True)
            n_doc = _copy_dataframe_in_batches(
                conn,
                df_doc,
                label="document",
                batch_size=DOCUMENT_COPY_BATCH_SIZE,
                copy_fn=_copy_documents,
            )
            print("document 已全部提交。", flush=True)

            print(f"COPY chunk（{len(df_chunk)} 行, dim={exp_dim}, 每批 {CHUNK_COPY_BATCH_SIZE}）…", flush=True)
            n_chunk = _copy_dataframe_in_batches(
                conn,
                df_chunk,
                label="chunk",
                batch_size=CHUNK_COPY_BATCH_SIZE,
                copy_fn=partial(_copy_chunks, expect_dim=exp_dim),
            )
            print("chunk 已全部提交。", flush=True)

            with conn.cursor() as cur:
                print("对齐 serial 序列 …", flush=True)
                _setval_sequences(cur)
            conn.commit()
    finally:
        tunnel.stop()

    print("完成并已提交。", flush=True)
    return n_doc, n_chunk


def load_parquets_via_tunnel_from_env(
    documents_parquet: Path | str | None = None,
    chunks_parquet: Path | str | None = None,
    *,
    truncate: bool | None = None,
    expect_embedding_dim: int | None = None,
) -> tuple[int, int]:
    """从环境变量读隧道与库配置；路径默认 ``ISKRA_DOCUMENTS_PARQUET`` / ``ISKRA_CHUNKS_PARQUET``。"""
    root = Path(__file__).resolve().parents[2]
    dp = Path(documents_parquet) if documents_parquet else None
    cp = Path(chunks_parquet) if chunks_parquet else None
    if dp is None:
        raw = os.environ.get("ISKRA_DOCUMENTS_PARQUET", "").strip()
        dp = Path(raw) if raw else root / "out" / "documents.parquet"
    if cp is None:
        raw = os.environ.get("ISKRA_CHUNKS_PARQUET", "").strip()
        cp = Path(raw) if raw else root / "out" / "chunks.parquet"

    trunc = truncate if truncate is not None else os.environ.get("ISKRA_LOAD_TRUNCATE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    return load_parquets_via_tunnel(
        dp,
        cp,
        tunnel_cfg=ssh_tunnel_config_from_env(),
        pg_cfg=pg_conn_config_from_env(),
        truncate=trunc,
        expect_embedding_dim=expect_embedding_dim,
    )
