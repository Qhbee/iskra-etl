"""loader：纯函数与 conninfo（不连真实 SSH/PG）。"""
from __future__ import annotations

import unittest

import numpy as np

from iskra_etl.loader import (
    _embedding_to_list,
    _embedding_to_pgvector_copy_literal,
    _tokenized_to_tsvector_copy_literal,
    build_conninfo_localhost,
    PgConnConfig,
)


class TestLoaderHelpers(unittest.TestCase):
    def test_embedding_to_list(self) -> None:
        v = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        self.assertEqual(_embedding_to_list(v, expect_dim=3), [0.0, 1.0, 2.0])

    def test_embedding_dim_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            _embedding_to_list([0.0, 1.0], expect_dim=3)

    def test_pgvector_copy_literal_bracket_not_brace(self) -> None:
        s = _embedding_to_pgvector_copy_literal([0.1, -0.2, 3.0])
        self.assertTrue(s.startswith("["), s)
        self.assertTrue(s.endswith("]"), s)
        self.assertNotIn("{", s)

    def test_tokenized_to_tsvector_copy_literal(self) -> None:
        lit = _tokenized_to_tsvector_copy_literal("马克思主义 哲学 基本 问题")
        self.assertIn("'马克思主义':1", lit)
        self.assertIn("'哲学':2", lit)
        self.assertIn("'基本':3", lit)
        self.assertIn("'问题':4", lit)
        self.assertEqual(_tokenized_to_tsvector_copy_literal(""), "")

    def test_tokenized_to_tsvector_escapes_single_quote(self) -> None:
        lit = _tokenized_to_tsvector_copy_literal("it's")
        self.assertIn("'it''s':1", lit)

    def test_build_conninfo_localhost(self) -> None:
        s = build_conninfo_localhost(
            65432,
            PgConnConfig(user="u", password="p", dbname="d", sslmode="disable"),
        )
        self.assertIn("127.0.0.1", s)
        self.assertIn("65432", s)


if __name__ == "__main__":
    unittest.main()
