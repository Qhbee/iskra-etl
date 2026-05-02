"""Embedder：L2 归一化、自适应 batch 阶梯、JSONL 流式列对齐；CUDA 可选多轮冒烟。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

# 轻量模型，测试不依赖 Jina 1024 权重下载（CI/首次环境友好）
_MINI_MODEL = "all-MiniLM-L6-v2"


class TestEmbedder(unittest.TestCase):
    def test_encode_texts_l2_unit_row_norms(self) -> None:
        from iskra_etl.embedder import encode_texts, load_sentence_model, resolve_embed_device

        model = load_sentence_model(model_id=_MINI_MODEL, device=resolve_embed_device("cpu"))
        texts = ["hello world", "sentence transformers", "numpy"]
        out = encode_texts(texts, model, batch_size=8, normalize_embeddings=True, document_prefix="")
        self.assertEqual(out.shape, (3, model.get_embedding_dimension()))
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, np.ones(3), atol=1e-4)

    def test_embed_chunk_records_order_preserved(self) -> None:
        from iskra_etl.embedder import embed_chunk_records, load_sentence_model, resolve_embed_device
        from iskra_etl.splitter import ChunkRecord

        model = load_sentence_model(model_id=_MINI_MODEL, device=resolve_embed_device("cpu"))
        recs = [
            ChunkRecord("a/x.md", 0, "alpha"),
            ChunkRecord("a/x.md", 1, "beta gamma"),
        ]
        embs = embed_chunk_records(recs, model, normalize_embeddings=False, document_prefix="")
        self.assertEqual(embs.shape[0], 2)
        self.assertFalse(np.allclose(embs[0], embs[1], rtol=0, atol=1e-6))

    def test_validate_dim_mismatch_raises(self) -> None:
        from iskra_etl.embedder import load_sentence_model, resolve_embed_device, validate_embedding_dim

        model = load_sentence_model(model_id=_MINI_MODEL, device=resolve_embed_device("cpu"))
        d = model.get_embedding_dimension()
        with self.assertRaises(ValueError):
            validate_embedding_dim(model, expected=d + 999)

    def test_iter_chunk_records_from_jsonl(self) -> None:
        from iskra_etl.embedder import embed_chunk_records, iter_chunk_records_from_jsonl, load_sentence_model, resolve_embed_device

        payload = [
            {"rel_path": "b/i.md", "chunk_index": 0, "chunk_text": "t0"},
            {"rel_path": "b/i.md", "chunk_index": 1, "chunk_text": "t1"},
        ]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.jsonl"
            with p.open("w", encoding="utf-8") as f:
                for row in payload:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            rows = list(iter_chunk_records_from_jsonl(p))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].rel_path, "b/i.md")
        self.assertEqual(rows[1].chunk_index, 1)

        model = load_sentence_model(model_id=_MINI_MODEL, device=resolve_embed_device("cpu"))
        embs = embed_chunk_records(rows, model, batch_size=2, document_prefix="")
        self.assertEqual(embs.shape, (2, model.get_embedding_dimension()))

    def test_prefix_affects_embedding(self) -> None:
        from iskra_etl.embedder import encode_texts, load_sentence_model, resolve_embed_device

        model = load_sentence_model(model_id=_MINI_MODEL, device=resolve_embed_device("cpu"))
        texts = ["hello world"]
        raw = encode_texts(texts, model, document_prefix="", normalize_embeddings=False)
        prefixed = encode_texts(texts, model, document_prefix="Document: ", normalize_embeddings=False)
        self.assertFalse(np.allclose(raw[0], prefixed[0], rtol=0, atol=1e-5))


class TestLowerBatchSize(unittest.TestCase):
    def test_prefers_ladder_rung(self) -> None:
        from iskra_etl.embedder import lower_batch_size_after_oom

        self.assertEqual(lower_batch_size_after_oom(48, 4), 32)
        self.assertEqual(lower_batch_size_after_oom(25, 4), 24)
        self.assertEqual(lower_batch_size_after_oom(5, 4), 4)

    def test_halves_when_below_ladder(self) -> None:
        from iskra_etl.embedder import lower_batch_size_after_oom

        self.assertEqual(lower_batch_size_after_oom(3, 1), 2)
        self.assertEqual(lower_batch_size_after_oom(2, 1), 1)


@unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
class TestCudaSustainedEmbed(unittest.TestCase):
    """多轮相同 batch，排除「越跑越爆」的常见路径（非严格内存泄漏检测）。"""

    def test_long_texts_batch48_many_rounds_minilm(self) -> None:
        """字符长约 2000、batch 48，多轮 encode；用轻量模型减轻下载与耗时。"""
        from iskra_etl.embedder import encode_texts, load_sentence_model

        model = load_sentence_model(model_id="all-MiniLM-L6-v2", device="cuda")
        line = "测" * 2000
        batch = [f"{line}:{i}" for i in range(48)]
        rounds = 15
        for _ in range(rounds):
            out = encode_texts(
                batch,
                model,
                batch_size=48,
                adaptive_batch_on_oom=True,
                show_progress_bar=False,
                document_prefix="",
            )
            self.assertEqual(out.shape, (48, model.get_embedding_dimension()))


if __name__ == "__main__":
    unittest.main()
