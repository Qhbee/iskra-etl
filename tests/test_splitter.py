"""LlamaIndex 切段：`chunk_index` 每篇从零递增；`rel_path` 为 posix 相对路径。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from iskra_etl.splitter import split_corpus_to_chunks

class TestSplitter(unittest.TestCase):
    def test_chunk_index_resets_per_document(self) -> None:

        # MarkdownNodeParser 按标题切：单段大正文几乎只有 1 块，需多节标题才有多块。
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

            records = list(split_corpus_to_chunks(root))

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
