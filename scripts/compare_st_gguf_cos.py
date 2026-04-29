"""对比 Sentence-Transformers（HF Jina）与 iskra-engine GGUF（llama.cpp）的余弦相似度。

- ST：`normalize_embeddings=True`（已是单位向量）。
- GGUF：`emit_gguf_batch.py` 子进程对每个文本做全长 L2 后再比较。

需在 iskra-etl 根目录配置 `.env`（ISKRA_EMBED_*）；在 iskra-engine 根目录配置 `ISKRA_GGUF_PATH`。

用法示例（PowerShell）：

  cd F:\\Projects\\PythonProjects\\iskra-etl
  $env:ISKRA_ENGINE_PYTHON=\"F:\\Projects\\PythonProjects\\iskra-engine\\.venv\\Scripts\\python.exe\"
  uv run python scripts\\compare_st_gguf_cos.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

_ETL_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ETL_ROOT / ".env")
_DEFAULT_ENGINE_ROOT = _ETL_ROOT.parent / "iskra-engine"
if (_DEFAULT_ENGINE_ROOT / ".env").is_file():
    load_dotenv(_DEFAULT_ENGINE_ROOT / ".env", override=False)

EXPECTED_DIM = 1024

# （name，document 侧字符串，query 侧字符串）；前缀按常见检索习惯：Document:/Query:
_SCENARIOS: list[tuple[str, str, str]] = [
    (
        "smoke对齐",
        "Document: 这是一条测试文本，用于确认向量维度。",
        "Query: 这句在测向量维度对吗？",
    ),
    (
        "英_天气复述",
        "Document: The weather is lovely today.",
        "Query: Is it sunny outside?",
    ),
    (
        "中_知识库片段",
        "Document: pgvector 在 PostgreSQL 里用半精度或可配置维度存向量。",
        "Query: PostgreSQL 怎么存向量？",
    ),
    (
        "同句_doc与query完全一致",
        "Document: 仅用于自检：同源同串。",
        "Document: 仅用于自检：同源同串。",
    ),
]


def _cosine_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """两端均为单位向量时等于点积。"""
    return float(np.dot(a, b))


def _infer_device() -> str:
    dev = os.environ.get("ISKRA_EMBED_DEVICE", "").strip().lower()
    if dev in ("cpu", "cuda"):
        return dev
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine-python",
        default=os.environ.get("ISKRA_ENGINE_PYTHON", ""),
        help="iskra-engine 的 .venv Python 可执行文件；或通过环境变量 ISKRA_ENGINE_PYTHON",
    )
    parser.add_argument(
        "--engine-root",
        default=os.environ.get(
            "ISKRA_ENGINE_ROOT", str(_DEFAULT_ENGINE_ROOT)
        ),
        help="iskra-engine 仓库根路径（内含 scripts/emit_gguf_batch.py）",
    )
    args = parser.parse_args()

    engine_py = (args.engine_python or "").strip()
    if not engine_py or not Path(engine_py).is_file():
        print(
            "请设置 --engine-python 或 ISKRA_ENGINE_PYTHON 指向 iskra-engine/.venv/Scripts/python.exe",
            file=sys.stderr,
        )
        sys.exit(1)

    engine_root = Path(args.engine_root).resolve()
    if (engine_root / ".env").is_file():
        load_dotenv(engine_root / ".env", override=False)

    emit_script = engine_root / "scripts" / "emit_gguf_batch.py"
    if not emit_script.is_file():
        print(f"找不到脚本: {emit_script}", file=sys.stderr)
        sys.exit(1)

    model_id = os.environ.get(
        "ISKRA_EMBED_MODEL",
        "jinaai/jina-embeddings-v5-text-small-retrieval",
    ).strip()
    device = _infer_device()
    print(f"ST loading {model_id!r} device={device!r} …", file=sys.stderr)
    model = SentenceTransformer(model_id, device=device)

    # 收集扁平文本列表以保证与 GGUF batch 索引一致：每场景 2 条（document, query）
    flat_texts: list[str] = []
    index_map: list[tuple[int, int]] = []  # (doc_flat_idx, q_flat_idx)

    offset = 0
    for _name, doc, qry in _SCENARIOS:
        index_map.append((offset, offset + 1))
        flat_texts.extend([doc, qry])
        offset += 2

    # --- ST ---
    emb_st = model.encode(
        flat_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if emb_st.shape != (len(flat_texts), EXPECTED_DIM):
        got = tuple(int(x) for x in emb_st.shape)
        print(
            f"ST 嵌入形状异常: {got}，期望 (_, {EXPECTED_DIM})",
            file=sys.stderr,
        )
        sys.exit(2)

    # --- GGUF 子进程（单次加载） ---
    td = tempfile.mkdtemp(prefix="iskra-gguf-")
    payload_path = Path(td) / "payload.json"
    payload_path.write_text(
        json.dumps({"texts": flat_texts}, ensure_ascii=False),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [engine_py, str(emit_script), "--input", str(payload_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload_path.unlink(missing_ok=True)
    Path(td).rmdir()

    if proc.returncode != 0:
        print(proc.stderr or proc.stdout, file=sys.stderr)
        sys.exit(proc.returncode or 1)

    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    unit_gguf = np.asarray(payload["unit_vectors"], dtype=np.float64)
    l2_before = payload["l2_before_norm"]
    if unit_gguf.shape != emb_st.shape:
        print(f"GGUF 形状 {unit_gguf.shape} 与 ST {emb_st.shape} 不一致", file=sys.stderr)
        sys.exit(2)

    # 校验单位长度
    for i, row in enumerate(unit_gguf):
        n = np.linalg.norm(row)
        if abs(n - 1.0) > 5e-3:
            print(
                f"警告: GGUF[{i}] ||v||≈{n:.6f}（应≈1）",
                file=sys.stderr,
            )

    print(
        "",
        "=" * 88,
        "\nScenario | cos(ST_doc,ST_query) | cos(GG_doc,GG_query) | cos(ST_doc,GG_doc) | cos(ST_q,GG_q) | GGUF L2(raw) doc/q",
        "\n",
        sep="",
    )

    emb_st_np = emb_st.astype(np.float64)

    for (name, _d, _q), (idi, idq) in zip(_SCENARIOS, index_map, strict=True):
        st_d = emb_st_np[idi]
        st_q = emb_st_np[idq]
        gg_d = unit_gguf[idi]
        gg_q = unit_gguf[idq]

        c_st = _cosine_numpy(st_d, st_q)
        c_gg = _cosine_numpy(gg_d, gg_q)
        c_dd = _cosine_numpy(st_d, gg_d)
        c_qq = _cosine_numpy(st_q, gg_q)

        lb_d = l2_before[idi]
        lb_q = l2_before[idq]

        print(
            f"{name:22s}"
            f" | {c_st:+.5f}"
            f" | {c_gg:+.5f}"
            f" | {c_dd:+.5f}"
            f" | {c_qq:+.5f}"
            f" | {lb_d:.4f}/{lb_q:.4f}"
        )

    print("说明：前两列为「各管线内部」文档-查询语义相关度；中间两列为两条管线对齐同一前缀文本时的向量接近度。", file=sys.stderr)


if __name__ == "__main__":
    main()
