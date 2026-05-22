"""Tokenizer：停用词、JSONL/txt 分词写出与行对齐。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from iskra_etl.tokenizer import (
    DEFAULT_STOPWORDS,
    load_chunk_texts_from_jsonl,
    tokenize_for_search,
    tokenize_jsonl_to_txt,
    write_chunks_tokenized_txt,
)


def _read_tokenized_txt_lines(path: Path) -> list[str]:
    """测试内读 ``chunks_tokenized.txt`` 行（不依赖 exporter）。"""
    with path.open(encoding="utf-8") as f:
        return [line.rstrip("\n\r") for line in f]


class TestTokenizer(unittest.TestCase):
    def test_tokenize_for_search_removes_stopwords(self) -> None:
        raw = "马克思主义哲学的基本问题"
        out = tokenize_for_search(raw, stopwords=DEFAULT_STOPWORDS)
        self.assertNotIn("的", out.split())
        self.assertIn("马克思主义", out.split())
        self.assertIn("哲学", out.split())
        self.assertIn("基本", out.split())
        self.assertIn("问题", out.split())

    def test_tokenize_empty_and_whitespace(self) -> None:
        self.assertEqual(tokenize_for_search(""), "")
        self.assertEqual(tokenize_for_search("   \n\t  "), "")

    def test_normalize_newlines_in_chunk(self) -> None:
        out = tokenize_for_search("第一行\n第二行", stopwords=frozenset())
        self.assertNotIn("\n", out)
        self.assertGreater(len(out.split()), 0)

    def test_load_chunk_texts_from_jsonl(self) -> None:
        payload = [
            {"rel_path": "a.md", "chunk_index": 0, "chunk_text": "甲"},
            {"rel_path": "a.md", "chunk_index": 1, "chunk_text": "乙"},
        ]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.jsonl"
            with p.open("w", encoding="utf-8") as f:
                for row in payload:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.assertEqual(load_chunk_texts_from_jsonl(p), ["甲", "乙"])

    def test_jsonl_txt_line_alignment(self) -> None:
        payload = [
            {"rel_path": "a/x.md", "chunk_index": 0, "chunk_text": "马克思主义哲学的基本问题"},
            {"rel_path": "a/x.md", "chunk_index": 1, "chunk_text": "列宁全集"},
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            jsonl = root / "chunks.jsonl"
            txt = root / "chunks_tokenized.txt"
            with jsonl.open("w", encoding="utf-8") as f:
                for row in payload:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            n = tokenize_jsonl_to_txt(jsonl, txt)
            self.assertEqual(n, 2)

            lines = _read_tokenized_txt_lines(txt)
            self.assertEqual(len(lines), 2)
            self.assertNotIn("的", lines[0].split())
            self.assertIn("列宁", lines[1])
            self.assertIn("全集", lines[1])

    def test_write_chunks_tokenized_txt_from_chunk_texts(self) -> None:
        texts = ["测试文本", ""]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "out.txt"
            n = write_chunks_tokenized_txt(p, texts, stopwords=frozenset())
            self.assertEqual(n, 2)
            lines = _read_tokenized_txt_lines(p)
            self.assertEqual(len(lines), 2)
            self.assertGreater(len(lines[0]), 0)
            self.assertEqual(lines[1], "")

    def test_max_chunks_limits_output_lines(self) -> None:
        payload = [
            {"rel_path": "a.md", "chunk_index": 0, "chunk_text": "甲"},
            {"rel_path": "a.md", "chunk_index": 1, "chunk_text": "乙"},
            {"rel_path": "a.md", "chunk_index": 2, "chunk_text": "丙"},
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            jsonl = root / "c.jsonl"
            txt = root / "c.txt"
            with jsonl.open("w", encoding="utf-8") as f:
                for row in payload:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n = tokenize_jsonl_to_txt(jsonl, txt, max_chunks=2)
            self.assertEqual(n, 2)
            self.assertEqual(len(_read_tokenized_txt_lines(txt)), 2)


if __name__ == "__main__":
    unittest.main()
