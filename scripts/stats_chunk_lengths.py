"""从 chunks.jsonl 统计每块 ``chunk_text`` 的长度分布（Unicode 字符数，与 Python ``len()`` 一致）。"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path


def _percentile(sorted_vals: list[int], q: float) -> float:
    """q  ∈ [0,100]，对已排序整数列表做线性插值分位数。"""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    k = (n - 1) * (q / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo] + (k - lo) * (sorted_vals[hi] - sorted_vals[lo]))


def _percentile_index(n: int, q: float) -> int:
    """q ∈ [0,100]，返回最接近该分位位置的样本下标（0-based）。"""
    if n <= 1:
        return 0
    k = (n - 1) * (q / 100.0)
    idx = int(round(k))
    if idx < 0:
        return 0
    if idx >= n:
        return n - 1
    return idx


def _preview_text(text: str, limit: int = 80) -> str:
    """单行预览：换行压成空格，过长则截断。"""
    one_line = " ".join(text.splitlines()).strip()
    if len(one_line) <= limit:
        return one_line
    return one_line[:limit] + "..."


def main() -> None:
    ap = argparse.ArgumentParser(description="统计 JSONL 各 chunk_text 的长度")
    ap.add_argument(
        "jsonl",
        nargs="?",
        type=Path,
        default=None,
        help="chunks.jsonl 路径（默认 env ISKRA_CHUNK_JSONL，再默认仓库根下 out/chunks.jsonl）",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    path = args.jsonl
    if path is None:
        raw = os.environ.get("ISKRA_CHUNK_JSONL", "").strip()
        path = Path(raw) if raw else root / "out" / "chunks.jsonl"
    path = path.resolve()

    if not path.is_file():
        print(f"文件不存在: {path}", file=sys.stderr)
        sys.exit(2)

    rows: list[tuple[int, str]] = []
    n_bad = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            text = obj.get("chunk_text")
            if text is None:
                n_bad += 1
                continue
            rows.append((len(text), str(text)))

    n = len(rows)
    if n == 0:
        print(f"无可统计行（坏行或未含 chunk_text: {n_bad}）路径={path}")
        sys.exit(1)

    lengths = [x[0] for x in rows]
    s = sorted(lengths)
    rows_sorted = sorted(rows, key=lambda x: x[0])
    mean_v = statistics.mean(lengths)
    median_v = statistics.median(lengths)

    lines_out = [
        f"文件: {path}",
        f"块数量: {n}" + (f"  （跳过 {n_bad} 行）" if n_bad else ""),
        "",
        "长度 = chunk_text 的 Unicode 字符数 len()（不等价于 tokenizer 口径）",
        f"最小: {min(lengths)}",
        f"最大: {max(lengths)}",
        f"平均: {mean_v:.2f}",
        f"中位数: {median_v:.2f}",
    ]
    if n > 1:
        lines_out.append(f"标准差: {statistics.stdev(lengths):.2f}")
    for p in (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99):
        lines_out.append(f"P{p}: {_percentile(s, float(p)):.0f}")
        ex_idx = _percentile_index(len(rows_sorted), float(p))
        ex_len, ex_text = rows_sorted[ex_idx]
        lines_out.append(f"  例子(len={ex_len}): {_preview_text(ex_text)}")

    print("\n".join(lines_out))


if __name__ == "__main__":
    main()
