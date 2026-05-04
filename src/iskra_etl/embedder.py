"""Embedder：Sentence-Transformers 批量编码（CUDA/CPU），内存中为 ``numpy`` 行向量。

``normalize_embeddings=True`` 与 pgvector 余弦（``<=>``）惯例一致，维数须与
``iskra-engine/sql/001_init.sql`` 中 ``vector(N)`` 一致（当前为 1024 + Jina small retrieval）。

**Jina retrieval**：送进 ``model.encode`` 前需要为每条文本拼接 ``Document: `` 前缀（**不改** JSONL 里的 ``chunk_text``），
由环境变量 ``ISKRA_EMBED_DOCUMENT_PREFIX`` 控制；未设置时默认 ``Document: ``，设为空串则关闭。

长句 + 大 batch 易 OOM；``encode_texts`` 在 CUDA 上默认 **遇 OOM 自动缩小本窗口 batch**（48→32→…），
避免整 job 因个别窗口失败。
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from iskra_etl.splitter import ChunkRecord

_DEFAULT_MODEL = "jinaai/jina-embeddings-v5-text-small-retrieval"

# CUDA OOM 时依次尝试更小的窗口 batch（再不行则对半当前窗口）。
_BATCH_FALLBACK_LADDER: tuple[int, ...] = (
    128, 96,
    64, 48,
    32, 24,
    16, 12,
    8, 6,
    4, 3,
    2,
    1,
)


def lower_batch_size_after_oom(attempted_window_size: int, min_batch_size: int) -> int:
    """返回严格更小的窗口 batch；优先踩梯子，否则折半。"""
    for step in sorted(_BATCH_FALLBACK_LADDER, reverse=True):
        if step < attempted_window_size and step >= min_batch_size:
            return step
    return max(min_batch_size, attempted_window_size // 2)


def _model_on_cuda(model: SentenceTransformer) -> bool:
    try:
        return next(model.parameters()).is_cuda
    except StopIteration:
        return False


def _cuda_empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_embed_device(explicit: str | None = None) -> str:
    """返回 ``"cuda"`` 或 ``"cpu"``。环境变量 ``ISKRA_EMBED_DEVICE`` 优先于自动探测。"""
    raw = (explicit if explicit is not None else os.environ.get("ISKRA_EMBED_DEVICE", "")).strip().lower()
    if raw in ("cpu", "cuda"):
        return raw
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_sentence_model(
    *,
    model_id: str | None = None,
    device: str | None = None,
) -> SentenceTransformer:
    mid = (model_id or os.environ.get("ISKRA_EMBED_MODEL", _DEFAULT_MODEL)).strip()
    dev = resolve_embed_device(device)
    return SentenceTransformer(mid, device=dev)


def expected_embedding_dim(from_env: bool = True) -> int | None:
    """若设置 ``ISKRA_EMBED_DIM`` 则解析为整数；否则 ``None``（不做维数断言）。"""
    raw = os.environ.get("ISKRA_EMBED_DIM", "").strip() if from_env else ""
    if not raw:
        return None
    return int(raw)


def validate_embedding_dim(model: SentenceTransformer, expected: int | None = None) -> int:
    """校验 ``model`` 输出维数是否与 ``expected`` 或 ``ISKRA_EMBED_DIM`` 一致。"""
    got = int(model.get_embedding_dimension())
    exp = expected if expected is not None else expected_embedding_dim(from_env=True)
    if exp is not None and got != exp:
        msg = (
            f"模型输出维数 {got} 与 ISKRA_EMBED_DIM/期望 {exp} 不一致；"
            "须与 001_init.sql 中 vector(N) 对齐。"
        )
        raise ValueError(msg)
    return got


def default_encode_batch_size() -> int:
    raw = os.environ.get("ISKRA_EMBED_BATCH_SIZE", "").strip()
    return int(raw) if raw else 48


def encode_texts(
    texts: Sequence[str],
    model: SentenceTransformer,
    *,
    batch_size: int | None = None,
    normalize_embeddings: bool = True,
    show_progress_bar: bool = False,
    adaptive_batch_on_oom: bool = True,
    min_batch_size: int = 4,
    document_prefix: str | None = None,
) -> np.ndarray:
    """对 ``texts`` 批量编码，返回 ``float32``、形状 ``(len(texts), dim)``的二维数组。

    ``document_prefix``：``None`` 时读 ``ISKRA_EMBED_DOCUMENT_PREFIX``（未设置则 ``"Document: "``）；仅影响传入模型的字符串，
    不修改调用方传入的 ``texts`` 原列表。

    CUDA 上默认 **窗口式** 编码：遇 ``torch.cuda.OutOfMemoryError`` 时将 **当前窗口**
    的 batch 沿阶梯降级（如 48→32→24）或折半重试，成功后下一窗口再恢复为 ``batch_size``。
    单条即 OOM 仍会抛错（需换模型 / 设备或截断输入）。

    ``min_batch_size`` 为自动降级下限（不低于 1 时 OOM 会尝试 1）。

    **进度条**：本函数若 ``show_progress_bar=True``，使用 **自建 tqdm** 进度条：
    （编码:   30%|███       | 24000/78984 [2:45:43<9:06:59,  2.40txt/s]。）
    不在此路径下启用 sentence-transformers 自带的 ``model.encode(..., show_progress_bar=True)`` 那种进度条：
    （Batches:   60%|██████    | 988/1646 [7:43:48<12:52:46, 18.32s/it]）
    库内置条按「整次 encode 固定 batch」假设绘制，但我们采用 **自适应 batch**，同一轮推理里各窗口的 batch 大小不固定，
    故 **一律** ``show_progress_bar=False``，仅由外层按「已处理文本条数」更新一条进度。
    """
    if not texts:
        d = int(model.get_embedding_dimension())
        return np.zeros((0, d), dtype=np.float32)

    target_bs = batch_size if batch_size is not None else default_encode_batch_size()
    target_bs = max(int(target_bs), 1)
    min_bs = max(1, int(min_batch_size))

    if document_prefix is None:
        document_prefix = os.environ.get("ISKRA_EMBED_DOCUMENT_PREFIX", "Document: ")
    texts_list = [f"{document_prefix}{t}" if document_prefix else t for t in texts]
    n = len(texts_list)
    use_adaptive = adaptive_batch_on_oom and _model_on_cuda(model)
    current_bs = target_bs

    parts: list[np.ndarray] = []
    start = 0

    # 原版（库内置）：model.encode(..., show_progress_bar=True) → 常见 tqdm 描述「Batches」，
    # total≈ceil(n/batch_size)。本处不用，因为采用了自适应 batch，OOM 时动态缩小 batch_size。
    pbar = None
    if show_progress_bar:
        try:
            from tqdm.auto import tqdm as tqdm_factory
        except ImportError:
            tqdm_factory = None
        if tqdm_factory is not None:
            pbar = tqdm_factory(total=n, unit="txt", desc="编码")

    while start < n:
        end = min(start + current_bs, n)
        batch = texts_list[start:end]
        bsz = len(batch)
        try:
            # 禁用 sentence-transformers 内部「Batches」条；进度仅由上方 pbar 按「文本条数」累计。
            arr = model.encode(
                batch,
                batch_size=bsz,
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=False,
            )
            row = np.asarray(arr, dtype=np.float32)
            if row.ndim != 2:
                msg = f"encode 期望二维数组，得到 shape={row.shape}"
                raise RuntimeError(msg)
            parts.append(row)
            start = end
            if use_adaptive:
                current_bs = target_bs
            if pbar is not None:
                pbar.update(bsz)
        except torch.cuda.OutOfMemoryError as exc:
            if not use_adaptive:
                raise
            _cuda_empty_cache()
            if bsz <= 1:
                if pbar is not None:
                    pbar.close()
                msg = (
                    "CUDA OOM：当前窗口仅 1 条仍失败；请减小 ISKRA_EMBED_BATCH_SIZE、"
                    "缩短文本或换设备。"
                )
                raise RuntimeError(msg) from exc
            new_bs = lower_batch_size_after_oom(bsz, min_bs)
            if new_bs >= bsz:
                new_bs = max(min_bs, bsz // 2)
            current_bs = new_bs

    if pbar is not None:
        pbar.close()

    if not parts:
        return np.zeros((0, int(model.get_embedding_dimension())), dtype=np.float32)
    out = np.vstack(parts)
    if out.shape[0] != n:
        msg = f"内部批次合并后行数 {out.shape[0]} 与输入 {n} 不一致"
        raise RuntimeError(msg)
    return out


def embed_chunk_records(
    records: Sequence[ChunkRecord],
    model: SentenceTransformer,
    *,
    batch_size: int | None = None,
    normalize_embeddings: bool = True,
    show_progress_bar: bool = False,
    adaptive_batch_on_oom: bool = True,
    min_batch_size: int = 4,
    document_prefix: str | None = None,
) -> np.ndarray:
    """与 ``records`` 顺序一致；文本取自 ``chunk_text``（仅编码时加 ``document_prefix``）。"""
    texts = [r.chunk_text for r in records]
    return encode_texts(
        texts,
        model,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=show_progress_bar,
        adaptive_batch_on_oom=adaptive_batch_on_oom,
        min_batch_size=min_batch_size,
        document_prefix=document_prefix,
    )


@dataclass(frozen=True)
class EmbeddedBatch:
    """一批块及其矩阵（行与 ``records`` 一一对应）。"""

    records: list[ChunkRecord]
    embeddings: np.ndarray


def iter_embed_chunk_records(
    records: Iterable[ChunkRecord],
    model: SentenceTransformer,
    *,
    batch_size: int | None = None,
    normalize_embeddings: bool = True,
    show_progress_bar: bool = False,
    adaptive_batch_on_oom: bool = True,
    min_batch_size: int = 4,
    document_prefix: str | None = None,
) -> Iterator[EmbeddedBatch]:
    """流式：按批编码，便于大 JSONL 控制内存。"""
    bs = batch_size if batch_size is not None else default_encode_batch_size()
    buf: list[ChunkRecord] = []
    for r in records:
        buf.append(r)
        if len(buf) >= bs:
            embs = embed_chunk_records(
                buf,
                model,
                batch_size=len(buf),
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=show_progress_bar,
                adaptive_batch_on_oom=adaptive_batch_on_oom,
                min_batch_size=min_batch_size,
                document_prefix=document_prefix,
            )
            yield EmbeddedBatch(records=list(buf), embeddings=embs)
            buf = []
    if buf:
        embs = embed_chunk_records(
            buf,
            model,
            batch_size=len(buf),
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
            adaptive_batch_on_oom=adaptive_batch_on_oom,
            min_batch_size=min_batch_size,
            document_prefix=document_prefix,
        )
        yield EmbeddedBatch(records=list(buf), embeddings=embs)


def iter_chunk_records_from_jsonl(path: Path | str) -> Iterator[ChunkRecord]:
    """步 3 产出的 JSONL：每行 ``rel_path`` / ``chunk_index`` / ``chunk_text``。"""
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            yield ChunkRecord(
                rel_path=str(o["rel_path"]),
                chunk_index=int(o["chunk_index"]),
                chunk_text=str(o["chunk_text"]),
            )


def save_chunk_embeddings_npy(path: Path | str, embeddings: np.ndarray) -> None:
    """将 ``embeddings`` 存为 ``.npy``（``float32``、二维 ``(N, D)``）。

    与 ``iter_chunk_records_from_jsonl`` 产物的 **行顺序一一对应**：第 ``i`` 行 JSONL
    对应 ``embeddings[i]``。供 ``export_parquet`` / :mod:`iskra_etl.exporter` 组装 Parquet。
    读取侧见 :func:`iskra_etl.exporter.load_chunk_embeddings_npy`。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        msg = f"chunk embeddings 期望二维数组，shape={arr.shape}"
        raise ValueError(msg)
    np.save(p, arr, allow_pickle=False)


def embed_jsonl_batches(
    path: Path | str,
    model: SentenceTransformer,
    *,
    batch_size: int | None = None,
    normalize_embeddings: bool = True,
    show_progress_bar: bool = False,
    adaptive_batch_on_oom: bool = True,
    min_batch_size: int = 4,
    document_prefix: str | None = None,
) -> Iterator[EmbeddedBatch]:
    """自 JSONL 流式读出 ``ChunkRecord`` 并按批编码（不向量化 Iterator 整块驻留内存）。"""
    yield from iter_embed_chunk_records(
        iter_chunk_records_from_jsonl(path),
        model,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=show_progress_bar,
        adaptive_batch_on_oom=adaptive_batch_on_oom,
        min_batch_size=min_batch_size,
        document_prefix=document_prefix,
    )
