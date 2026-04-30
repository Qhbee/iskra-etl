"""LlamaIndex 切段：`chunk_index` 每篇从零递增；`rel_path` 为 posix 相对路径。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from iskra_etl.splitter import consolidate_small_fragments, normalize_newlines, split_corpus_to_chunks


class TestNormalizeNewlines(unittest.TestCase):
    def test_crlf_to_lf(self) -> None:
        self.assertEqual(normalize_newlines("a\r\n\r\nb"), "a\n\nb")
        self.assertEqual(normalize_newlines("x\ry"), "x\ny")


class TestConsolidateSmallFragments(unittest.TestCase):
    def test_merge_forward_until_over_threshold(self) -> None:
        pieces = ["a" * 200, "b" * 200, "c" * 200]
        out = consolidate_small_fragments(pieces, min_chars=512)
        self.assertEqual(len(out), 1)
        self.assertGreater(len(out[0]), 512)

    def test_tail_merges_into_previous(self) -> None:
        pieces = ["x" * 600, "y" * 50, "z" * 40]
        out = consolidate_small_fragments(pieces, min_chars=512)
        self.assertEqual(len(out), 1)
        self.assertIn("y" * 50, out[0])

    def test_single_short_document_stays_one(self) -> None:
        out = consolidate_small_fragments(["hi"], min_chars=512)
        self.assertEqual(out, ["hi"])


class TestSplitter(unittest.TestCase):
    def test_chunk_index_resets_per_document(self) -> None:
        """每篇内 chunk_index 从 0 递增；与生产默认无关，用显式参数保证「多篇多块」。"""
        # MarkdownNodeParser 按标题切：单段大正文几乎只有 1 块，需多节标题才有多块。
        # 这里小正文即可；若用 split_corpus 生产默认（如 consolidate_min_chars=512），短节会被缝成每篇 1 块，
        # 这是合理行为，不是实现 bug——故此处固定较小缝合阈值与句切上限，专门测 index 语义。
        def _md_with_sections(tag: str) -> str:
            parts: list[str] = [f"# Doc {tag}", ""]
            for i in range(4):
                parts.extend([f"## Sec-{tag}-{i}", "", "Line. " * 40, ""])
            return "\n".join(parts)

        long_body = _md_with_sections("a")
        long_body2 = _md_with_sections("b")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a").mkdir()
            (root / "b").mkdir()
            (root / "a" / "index.md").write_text(long_body, encoding="utf-8")
            (root / "b" / "index.md").write_text(long_body2, encoding="utf-8")

            records = list(
                split_corpus_to_chunks(
                    root,
                    sentence_chunk_size=512,
                    sentence_chunk_overlap=64,
                    consolidate_min_chars=128,
                ),
            )

        paths = [r.rel_path for r in records]
        self.assertGreater(len(records), 2)

        chunks_a = [r for r in records if r.rel_path.endswith("a/index.md")]
        chunks_b = [r for r in records if r.rel_path.endswith("b/index.md")]
        self.assertGreaterEqual(len(chunks_a), 2)
        self.assertGreaterEqual(len(chunks_b), 2)

        indices_a = [r.chunk_index for r in chunks_a]
        indices_b = [r.chunk_index for r in chunks_b]
        self.assertEqual(indices_a, list(range(len(chunks_a))))
        self.assertEqual(indices_b, list(range(len(chunks_b))))

        for p in paths:
            self.assertFalse(Path(p).is_absolute())

    def test_yaml_frontmatter_stripped_before_parse(self) -> None:
        fm = "---\r\nbook: corpus\r\ntitle: x\r\n---\r\n\r\n"
        body_md = fm + "## First\n\nHello.\r\n\r\n## Second\r\n\r\nWorld."
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "d").mkdir()
            (root / "d" / "index.md").write_text(body_md, encoding="utf-8")
            records = list(split_corpus_to_chunks(root))

        joined = "\n".join(r.chunk_text for r in records)
        self.assertNotIn("book: corpus", joined)
        self.assertFalse(records[0].chunk_text.lstrip().startswith("---"))


if __name__ == "__main__":
    unittest.main()
