# iskra-etl

[![Python](https://img.shields.io/badge/python-3.14-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Status](https://img.shields.io/badge/Status-Active-success)

---

`iskra-etl` 是 `Iskra` 项目的 ETL 部分（离线切块 + 向量化等），采用 [iskra-data](https://github.com/Qhbee/iskra-data) 的数据 。

最上游的语料一般由 [iskra-x2md](https://github.com/Qhbee/iskra-x2md) 从 EPUB/PDF 生成；
下游由 [iskra-engine](https://github.com/Qhbee/iskra-engine) 做向量检索与 RAG。

## `src/iskra_etl` 四个模块

| 模块 | 职责 |
|------|------|
| **`splitter`** | 扫描语料根下 `**/index.md`，去 frontmatter，LlamaIndex 按标题骨架 + token 切段，过短块合并；输出切块记录。 |
| **`embedder`** | 读切块 JSONL，用 Sentence-Transformers（默认 Jina 1024 维）批量编码，L2 归一化；编码前拼 `Document: ` 前缀。 |
| **`exporter`** | 从语料生成 `documents.parquet`；把 JSONL 与行对齐的 `.npy` 拼成 `chunks.parquet`（含 `document_id`、文本、向量）。 |
| **`loader`** | 笔记本经 SSH 隧道连远程 PG，对 `document` / `chunk` 两表做 `COPY`；非空库全量重载需 `--truncate`。 |

## `tests` 四个测试

与 `src/iskra_etl` 里四个模块的一一对应

## 流程与产物

按顺序跑 `scripts/` 下四个 CLI（路径可用 `.env` 覆盖，见 [.env.example](.env.example)）：

```mermaid
flowchart TB
  corpus(["iskra-data/**/*.md"])
  jsonl["chunks.jsonl"]
  npy["chunks_embeddings.npy"]
  docsPq["documents.parquet"]
  exportJoin(("﻿"))
  chunksPq["chunks.parquet"]
  loadJoin(("﻿"))
  db[("PostgreSQL: document + chunk")]

  corpus -->|"split_chunks.py (splitter)"| jsonl
  jsonl -->|"embed_chunks.py (embedder)"| npy
  corpus -->|"export_parquet.py (exporter)"| docsPq
  jsonl --> exportJoin
  npy --> exportJoin
  exportJoin -->|"export_parquet.py (exporter)"| chunksPq
  docsPq --> loadJoin
  chunksPq --> loadJoin
  loadJoin -->|"load_to_db.py (loader)"| db
```

**注意**：`chunks.jsonl` 与 `chunks_embeddings.npy` 必须同一次切块、同一次 embed，行数一致；改切块后需重跑 embed 及之后各步。

## 快速开始

```bash
cp .env.example .env   # 填语料根、SSH、PG 等
uv sync

uv run python scripts/split_chunks.py
uv run python scripts/embed_chunks.py
uv run python scripts/export_parquet.py
uv run python scripts/load_to_db.py --truncate   # 非空库全量重载时
```

其它脚本：`stats_chunk_lengths.py`（块长统计）、`smoke_st_embed.py` / `compare_st_gguf_cos.py`（嵌入冒烟与 ST/GGUF 对比），非主链路。
