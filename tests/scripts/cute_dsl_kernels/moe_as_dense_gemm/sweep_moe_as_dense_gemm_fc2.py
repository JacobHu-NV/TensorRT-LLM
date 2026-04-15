# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Sweep script for MoE DenseGEMM FC2 kernel over m, tactic, and split_k.

Usage:
  python sweep_moe_as_dense_gemm_fc2.py
  python sweep_moe_as_dense_gemm_fc2.py --output results.csv
  python sweep_moe_as_dense_gemm_fc2.py --nkl 7168,65536,1 --m_range 32,384,32
"""

import argparse
import csv
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import cutlass

# Import run() from the sibling script
sys.path.insert(0, str(Path(__file__).parent))
from run_moe_as_dense_gemm_fc2 import run  # noqa: E402

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
M_START, M_STOP, M_STEP = 32, 416, 32
M_VALUES = list(range(M_START, M_STOP, M_STEP))

TACTICS = [
    ((128, 128), (1, 1)),
    ((128, 128), (1, 2)),
    ((128, 256), (1, 2)),
    ((128, 64), (1, 1)),
    ((256, 128), (2, 1)),
]

SPLIT_K_VALUES = [1, 2, 4]

# Fixed problem dimensions (n, k, l)
DEFAULT_NKL = (7168, 65536, 1)
DEFAULT_EXPERT_COUNT = 256

# Benchmark settings
WARMUP_ITERATIONS = 10
ITERATIONS = 50


def write_pivot_csv(rows, pivot_path):
    """Write a pivot CSV: rows = m values, columns = (tactic, split_k) combos."""
    # Collect column keys in insertion order
    col_keys = []
    seen_keys = set()
    for r in rows:
        key = (
            (r["mma_tiler_m"], r["mma_tiler_n"]),
            (r["cluster_m"], r["cluster_n"]),
            r["split_k"],
        )
        if key not in seen_keys:
            seen_keys.add(key)
            col_keys.append(key)

    def col_label(key):
        mma, cluster, sk = key
        return f"({mma[0]}, {mma[1]}),({cluster[0]}, {cluster[1]}),{sk}"

    cell = {}
    for r in rows:
        key = (
            (r["mma_tiler_m"], r["mma_tiler_n"]),
            (r["cluster_m"], r["cluster_n"]),
            r["split_k"],
        )
        cell[(r["m"], key)] = r["exec_time_us"] if r["status"] == "PASS" else ""

    m_values = sorted({r["m"] for r in rows})

    with open(pivot_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["m"] + [col_label(k) for k in col_keys] + ["min_time", "min_time_config"])
        for m in m_values:
            times = [(cell[(m, k)], k) for k in col_keys if (m, k) in cell and cell[(m, k)] != ""]
            if times:
                min_val, min_key = min(times, key=lambda x: float(x[0]))
                min_time = min_val
                min_config = col_label(min_key)
            else:
                min_time = ""
                min_config = ""
            writer.writerow([m] + [cell.get((m, k), "") for k in col_keys] + [min_time, min_config])

    print(f"Pivot results written to {pivot_path}")


def sweep(nkl, expert_count, output_path, m_values=None):
    n, k, l = nkl  # noqa: E741
    if m_values is None:
        m_values = M_VALUES
    rows = []
    total = len(m_values) * len(TACTICS) * len(SPLIT_K_VALUES)
    done = 0

    for m in m_values:
        for mma_tiler_mn, cluster_shape_mn in TACTICS:
            for split_k in SPLIT_K_VALUES:
                done += 1
                tag = (
                    f"m={m:4d}  tactic={mma_tiler_mn}/{cluster_shape_mn}"
                    f"  split_k={split_k}  [{done}/{total}]"
                )
                try:
                    with redirect_stdout(io.StringIO()):
                        exec_time = run(
                            mnkl=(m, n, k, l),
                            expert_count=expert_count,
                            ab_dtype=cutlass.Float4E2M1FN,
                            sf_dtype=cutlass.Float8E8M0FNU,
                            sf_vec_size=16,
                            c_dtype=cutlass.Float16,
                            a_major="k",
                            b_major="k",
                            c_major="n",
                            mma_tiler_mn=mma_tiler_mn,
                            cluster_shape_mn=cluster_shape_mn,
                            warmup_iterations=WARMUP_ITERATIONS,
                            iterations=ITERATIONS,
                            skip_ref_check=False,
                            use_cold_l2=False,
                            use_cupti=True,
                            split_k=split_k,
                        )
                    print(f"PASS  {tag}  -> {exec_time:.2f} us")
                    rows.append(
                        {
                            "m": m,
                            "n": n,
                            "k": k,
                            "l": l,
                            "mma_tiler_m": mma_tiler_mn[0],
                            "mma_tiler_n": mma_tiler_mn[1],
                            "cluster_m": cluster_shape_mn[0],
                            "cluster_n": cluster_shape_mn[1],
                            "split_k": split_k,
                            "exec_time_us": f"{exec_time:.4f}",
                            "status": "PASS",
                        }
                    )
                except (TypeError, ValueError) as e:
                    print(f"SKIP  {tag}  ({e})")
                    rows.append(
                        {
                            "m": m,
                            "n": n,
                            "k": k,
                            "l": l,
                            "mma_tiler_m": mma_tiler_mn[0],
                            "mma_tiler_n": mma_tiler_mn[1],
                            "cluster_m": cluster_shape_mn[0],
                            "cluster_n": cluster_shape_mn[1],
                            "split_k": split_k,
                            "exec_time_us": "",
                            "status": f"SKIP: {e}",
                        }
                    )

    fieldnames = [
        "m",
        "n",
        "k",
        "l",
        "mma_tiler_m",
        "mma_tiler_n",
        "cluster_m",
        "cluster_n",
        "split_k",
        "exec_time_us",
        "status",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {output_path}")

    p = Path(output_path)
    pivot_path = p.with_name(p.stem + "_pivot" + p.suffix)
    write_pivot_csv(rows, pivot_path)


if __name__ == "__main__":

    def parse_ints(s):
        return tuple(int(x) for x in s.split(","))

    parser = argparse.ArgumentParser(description="Sweep MoE DenseGEMM FC2 over m/tactic/split_k.")
    parser.add_argument(
        "--nkl",
        type=parse_ints,
        default=DEFAULT_NKL,
        help="Fixed (n,k,l) dimensions (default: 7168,65536,1)",
    )
    parser.add_argument(
        "--expert_count", type=int, default=DEFAULT_EXPERT_COUNT, help="Expert count (default: 256)"
    )
    parser.add_argument(
        "--m_range",
        type=parse_ints,
        default=(M_START, M_STOP, M_STEP),
        help="m sweep as start,stop,step (default: 32,384,32)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sweep_fc2_results.csv",
        help="Output CSV file path (default: sweep_fc2_results.csv)",
    )
    args = parser.parse_args()

    m_values = list(range(args.m_range[0], args.m_range[1], args.m_range[2]))
    sweep(args.nkl, args.expert_count, args.output, m_values)
