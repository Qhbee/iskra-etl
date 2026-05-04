"""exporter：document_id 映射、Parquet 列、与 chunks 对齐。"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from iskra_etl.exporter import (
    build_documents_rows,
    document_id_by_rel_path,
    document_row_for_path,
    load_chunk_embeddings_npy,
    title_book_from_raw_markdown,
    write_chunks_parquet,
    write_chunks_parquet_from_jsonl_and_npy,
    write_documents_parquet,
)
from iskra_etl.splitter import split_corpus_to_chunks


class TestExporter(unittest.TestCase):
    def test_load_chunk_embeddings_npy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "e.npy"
            arr = np.arange(12, dtype=np.float32).reshape(3, 4)
            np.save(p, arr, allow_pickle=False)
            got = np.asarray(load_chunk_embeddings_npy(p, mmap_mode=None), dtype=np.float32)
            np.testing.assert_array_equal(got, arr)

    def test_document_id_order_follows_sorted_glob(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "z").mkdir()
            (root / "a").mkdir()
            (root / "z" / "index.md").write_text("# Z\n", encoding="utf-8")
            (root / "a" / "index.md").write_text("# A\n", encoding="utf-8")

            idmap = document_id_by_rel_path(root, id_start=1)
        self.assertEqual(idmap["a/index.md"], 1)
        self.assertEqual(idmap["z/index.md"], 2)

    def test_content_sha256_uses_body_without_frontmatter(self) -> None:
        raw = "---\ntitle: Hi\n---\n\n# Body\n\nOnly this."
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "x").mkdir()
            (root / "x" / "index.md").write_text(raw, encoding="utf-8")
            from iskra_etl.splitter import normalize_newlines, _strip_md_yaml_frontmatter

            body = normalize_newlines(_strip_md_yaml_frontmatter(raw))
            expected = hashlib.sha256(body.encode("utf-8")).hexdigest()
            row = document_row_for_path(root, "x/index.md", 7)
        self.assertEqual(row["content_sha256"], expected)
        self.assertIn("title: Hi", str(row["full_text"]))

    def test_title_book_from_frontmatter(self) -> None:
        raw = '---\ntitle: "My T"\nbook: vol1\n---\n\n# x\n'
        t, b = title_book_from_raw_markdown(raw)
        self.assertEqual(t, "My T")
        self.assertEqual(b, "vol1")

    def test_chunks_parquet_aligns_document_id(self) -> None:
        def _md(name: str) -> str:
            return "\n".join([f"# {name}", "", "## S0", "", "Hello. " * 30, "", "## S1", "", "World. " * 30])

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "b").mkdir()
            (root / "b" / "index.md").write_text(_md("b"), encoding="utf-8")
            idmap = document_id_by_rel_path(root)
            records = list(split_corpus_to_chunks(root))
            n = len(records)
            self.assertGreaterEqual(n, 1)
            emb = np.random.randn(n, 16).astype(np.float32)

            out = Path(td) / "c.parquet"
            write_chunks_parquet(
                document_id_by_rel=idmap,
                records=records,
                embeddings=emb,
                path=out,
                expect_dim=16,
            )
            df = pd.read_parquet(out)
        self.assertEqual(len(df), n)
        self.assertTrue((df["document_id"] == idmap["b/index.md"]).all())
        self.assertEqual(list(df["chunk_index"].values), [r.chunk_index for r in records])

    def test_write_chunks_parquet_from_jsonl_and_npy(self) -> None:
        import json

        def _md(name: str) -> str:
            return "\n".join([f"# {name}", "", "## S0", "", "Hello. " * 30])

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "b").mkdir()
            (root / "b" / "index.md").write_text(_md("b"), encoding="utf-8")
            idmap = document_id_by_rel_path(root)
            records = list(split_corpus_to_chunks(root))
            n = len(records)
            jpath = Path(td) / "x.jsonl"
            with jpath.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(
                        json.dumps(
                            {
                                "rel_path": r.rel_path,
                                "chunk_index": r.chunk_index,
                                "chunk_text": r.chunk_text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n",
                    )
            npyp = Path(td) / "e.npy"
            emb = np.random.randn(n, 16).astype(np.float32)
            np.save(npyp, emb, allow_pickle=False)

            outp = Path(td) / "ch.parquet"
            write_chunks_parquet_from_jsonl_and_npy(
                chunks_jsonl=jpath,
                embeddings_npy=npyp,
                document_id_by_rel=idmap,
                path=outp,
                expect_dim=16,
                mmap_npy=False,
            )
            df = pd.read_parquet(outp)
        self.assertEqual(len(df), n)

    def test_jsonl_npy_row_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            jpath = Path(td) / "j.jsonl"
            with jpath.open("w", encoding="utf-8") as f:
                f.write('{"rel_path":"a/x.md","chunk_index":0,"chunk_text":"x"}\n')
            np.save(Path(td) / "b.npy", np.zeros((2, 4), dtype=np.float32), allow_pickle=False)

            with self.assertRaises(ValueError):
                write_chunks_parquet_from_jsonl_and_npy(
                    chunks_jsonl=jpath,
                    embeddings_npy=Path(td) / "b.npy",
                    document_id_by_rel={"a/x.md": 1},
                    path=Path(td) / "o.parquet",
                    mmap_npy=False,
                )

    def test_write_documents_roundtrip(self) -> None:
        raw = "---\ntitle: T\n---\n\nbody"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "d" / "index.md").parent.mkdir(parents=True)
            (root / "d" / "index.md").write_text(raw, encoding="utf-8")
            idmap = document_id_by_rel_path(root)
            rows = build_documents_rows(root, idmap)
            outp = Path(td) / "d.parquet"
            write_documents_parquet(rows, outp)
            df = pd.read_parquet(outp)
        self.assertEqual(int(df.iloc[0]["id"]), 1)
        self.assertEqual(df.iloc[0]["title"], "T")


if __name__ == "__main__":
    unittest.main()
