"""用 HuggingFace 上的 Jina v5 small retrieval（Sentence-Transformers）试跑一条向量。

与 iskra-engine/scripts/smoke_gguf_embed.py 对齐：同一句子、同「Document:」前缀、期望 dim=1024。
注意：PyTorch 权重与 GGUF Q8 的浮点不必逐维相同，只要求维度与用法（retrieval 前缀）一致即可。

可在项目根目录 `.env` 配置（参见 `.env.example`），例如：ISKRA_EMBED_DEVICE=cuda
环境变量（可选）：
  ISKRA_EMBED_MODEL  默认 jinaai/jina-embeddings-v5-text-small-retrieval
  ISKRA_EMBED_DEVICE  强制 cpu 或 cuda；不设则自动 cuda 若可用
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

# 无论从哪启动，均加载「iskra-etl 根目录」下的 .env
_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")

EXPECTED_DIM = 1024


def main() -> None:

    model_id = os.environ.get(
        "ISKRA_EMBED_MODEL",
        "jinaai/jina-embeddings-v5-text-small-retrieval",
    ).strip()
    dev = os.environ.get("ISKRA_EMBED_DEVICE", "").strip().lower()

    if dev in ("cpu", "cuda"):
        device = dev
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"loading {model_id!r} on {device!r} ...", file=sys.stderr)
    model = SentenceTransformer(model_id, device=device)

    # 与 GGUF smoke 及 Jina retrieval 文档习惯一致
    text = "Document: 这是一条测试文本，用于确认向量维度。"
    emb = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vec = emb[0]
    dim = int(vec.shape[0])
    print(f"OK  dim={dim}  sentence-transformers  device={device}")
    if dim != EXPECTED_DIM:
        print(
            f"警告: 期望 {EXPECTED_DIM} 维（jina-v5-text-small-retrieval），当前 {dim}",
            file=sys.stderr,
        )
        sys.exit(2)
    print("前 5 维:", vec[:5].tolist())

    # HF 文档式：多句 + 相似度矩阵（与主流程无关，便于确认安装）
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]
    embs = model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    sim = util.cos_sim(embs, embs)
    print("similarities.shape:", tuple(sim.shape))

if __name__ == "__main__":
    main()
