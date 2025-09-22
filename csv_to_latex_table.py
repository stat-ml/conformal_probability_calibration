#!/usr/bin/env python3
"""
Generate a LaTeX table from a CSV with metrics like "mean ± std".

Rules:
- Bold and underline the BEST metric per column (entire formula, not just mean):
    * accuracy:          maximize (highest mean is best)
    * nll, brier_score, ece, mce, cw-ece, cmce: minimize (lowest mean is best)
- Coverage columns named like: coverage_[L, U]
    * Bold and underline if the mean is within [L, U] (inclusive), regardless of being best.
- Every numeric cell is rendered in math mode: $<value> \pm <value>$
- String columns (Dataset, Model, Calibrator) are printed as-is (with LaTeX escaping)
- Uses booktabs in the LaTeX table. You can wrap it in \resizebox if it's wide.

Usage:
    python gen_table.py --csv results.csv --out table.tex --caption "Evaluation results" --label tab:imagenet_mini
"""

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd


# -------- Helpers --------

PM_PATTERNS = [
    r"^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*±\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$",
    r"^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\+/-\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$",
]

def parse_pm(value: str) -> Optional[Tuple[float, float]]:
    """Parse 'mean ± std' or 'mean +/- std' into floats."""
    if value is None:
        return None
    s = str(value).strip()
    for pat in PM_PATTERNS:
        m = re.match(pat, s)
        if m:
            return float(m.group(1)), float(m.group(2))
    # allow single number (no std)
    try:
        v = float(s)
        return v, 0.0
    except Exception:
        return None

def latex_escape(text: str) -> str:
    """Escape LaTeX special characters commonly present in table text."""
    if text is None:
        return ""
    # Only escape the ones likely in your data; underscore is the big one.
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = str(text)
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out

COVERAGE_COL_RE = re.compile(r"^coverage_\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]\s*$")

def is_coverage_col(col: str) -> Optional[Tuple[float, float]]:
    """Return (low, high) if column is coverage_[low, high], else None."""
    m = COVERAGE_COL_RE.match(col)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None

def is_lower_better(col: str) -> bool:
    """Decide whether a metric should be minimized (True) or maximized (False)."""
    lower_better_names = {"nll", "brier_score", "ece", "mce", "cw-ece", "cmce"}
    # coverage columns are handled separately (not best-min/best-max)
    base = col.lower()
    if base in lower_better_names:
        return True
    return False

def is_higher_better(col: str) -> bool:
    """Decide whether a metric should be maximized (True)."""
    return col.lower() == "accuracy"

def find_best_indices(values: List[Optional[Tuple[float, float]]],
                      minimize: bool,
                      eps: float = 1e-12) -> List[int]:
    """Return indices of rows that are best (min or max of mean), ties included."""
    means = [v[0] if v else (math.inf if minimize else -math.inf) for v in values]
    if minimize:
        best = min(means)
        return [i for i, m in enumerate(means) if abs(m - best) <= eps]
    else:
        best = max(means)
        return [i for i, m in enumerate(means) if abs(m - best) <= eps]

def format_math(mean: float, std: float, bold: bool = False) -> str:
    """Format a number as LaTeX math with optional bold/underline for the whole formula."""
    formula = f"{mean:.4f} \\pm {std:.4f}"
    if bold:
        # Bold and underline the entire formula
        return r"$\underline{\mathbf{" + formula + r"}}$"
    else:
        return r"$" + formula + r"$"

# -------- Main generation --------

def generate_latex_table(df: pd.DataFrame,
                         caption: str,
                         label: str,
                         wrap_resizebox: bool = True) -> str:
    # Identify fixed text columns and metric columns
    # Expecting the first three text columns as in your data:
    text_cols = ["Dataset", "Model", "Calibrator"]
    # Allow for varying capitalization by mapping:
    lower_map = {c.lower(): c for c in df.columns}
    # Normalize presence
    for need in ["dataset", "model", "calibrator"]:
        if need not in lower_map:
            raise ValueError(f"Required column '{need}' not found in CSV.")
    text_cols = [lower_map["dataset"], lower_map["model"], lower_map["calibrator"]]

    metric_cols = [c for c in df.columns if c not in text_cols]

    # Pre-parse all metric cells into (mean, std) or None for non-numeric
    parsed: Dict[str, List[Optional[Tuple[float, float]]]] = {}
    for col in metric_cols:
        parsed[col] = [parse_pm(x) for x in df[col].tolist()]

    # Determine best rows for each metric (non-coverage)
    best_indices_by_col: Dict[str, List[int]] = {}
    for col in metric_cols:
        if is_coverage_col(col):
            continue  # handled via interval rule
        vals = parsed[col]
        if is_higher_better(col):
            idxs = find_best_indices(vals, minimize=False)
        elif is_lower_better(col):
            idxs = find_best_indices(vals, minimize=True)
        else:
            # If unknown metric, default to minimizing (conservative)
            idxs = find_best_indices(vals, minimize=True)
        best_indices_by_col[col] = idxs

    # Build LaTeX header
    header_cols = text_cols + metric_cols
    header_latex = " & ".join(latex_escape(c) for c in header_cols) + r" \\"

    # Column alignment: l l l then metrics centered
    alignment = "l l l " + " ".join(["c"] * len(metric_cols))

    # Pre-compute uncalibrated baselines per (Dataset, Model)
    def _is_uncalibrated(name: str) -> bool:
        s = str(name).strip().lower()
        return s in {"uncalibrated", "none"}

    def _group_key(i: int) -> tuple:
        return (str(df.at[i, text_cols[0]]), str(df.at[i, text_cols[1]]))

    baseline_index_by_group: Dict[tuple, int] = {}
    for i in range(len(df)):
        if _is_uncalibrated(df.at[i, text_cols[2]]):
            gk = _group_key(i)
            if gk not in baseline_index_by_group:
                baseline_index_by_group[gk] = i

    def _is_better(col: str, mean: float, base_mean: float) -> bool:
        if is_coverage_col(col):
            return False
        if is_higher_better(col):
            return mean > base_mean + 1e-12
        elif is_lower_better(col):
            return mean < base_mean - 1e-12
        # default minimize
        return mean < base_mean - 1e-12

    def _is_worse(col: str, mean: float, base_mean: float) -> bool:
        if is_coverage_col(col):
            return False
        if is_higher_better(col):
            return mean < base_mean - 1e-12
        elif is_lower_better(col):
            return mean > base_mean + 1e-12
        # default minimize
        return mean > base_mean + 1e-12

    # Build body rows with custom ordering and separators
    def _calibrator_group(name: str) -> int:
        s = str(name).strip().lower()
        if s in {"uncalibrated", "none"}:
            return 0
        if s.startswith("cnfrml"):
            return 2
        return 1

    indices = list(range(len(df)))
    ordered_indices = sorted(indices, key=lambda i: (_calibrator_group(df.at[i, text_cols[2]]), i))

    body_lines: List[str] = []
    prev_group: Optional[int] = None
    for i in ordered_indices:
        cur_group = _calibrator_group(df.at[i, text_cols[2]])
        if prev_group is not None and cur_group > prev_group:
            body_lines.append(r"\\")
        prev_group = cur_group
        row_cells: List[str] = []
        # text cells
        for col in text_cols:
            row_cells.append(latex_escape(df.at[i, col]))
        # metric cells
        for col in metric_cols:
            val = parsed[col][i]
            cov_bounds = is_coverage_col(col)
            if val is None:
                row_cells.append(latex_escape(str(df.at[i, col])))
                continue
            mean, std = val

            bold = False
            if cov_bounds is not None:
                lo, hi = cov_bounds
                # Bold/underline if mean in [lo, hi]
                if mean >= lo - 1e-12 and mean <= hi + 1e-12:
                    bold = True
            else:
                # Bold/underline if this row is a best index for the metric
                if i in best_indices_by_col.get(col, []):
                    bold = True

            cell = format_math(mean, std, bold=bold)

            # Optional green color if better than uncalibrated baseline (non-coverage metrics)
            gk = _group_key(i)
            base_idx = baseline_index_by_group.get(gk)
            if base_idx is not None and cov_bounds is None:
                base_val = parsed[col][base_idx]
                if base_val is not None:
                    base_mean, _ = base_val
                    if _is_better(col, mean, base_mean):
                        cell = r"\textcolor{green}{" + cell + r"}"
                    elif _is_worse(col, mean, base_mean):
                        cell = r"\textcolor{red}{" + cell + r"}"

            row_cells.append(cell)

        body_lines.append(" & ".join(row_cells) + r" \\")
    # Compose full table with booktabs
    table_core = []
    table_core.append(r"\begin{table*}[htbp]")
    table_core.append(r"\centering")
    if wrap_resizebox:
        table_core.append(r"\resizebox{\textwidth}{!}{%")
    table_core.append(r"\begin{tabular}{" + alignment + r"}")
    table_core.append(r"\toprule")
    table_core.append(header_latex)
    table_core.append(r"\midrule")
    table_core.extend(body_lines)
    table_core.append(r"\bottomrule")
    table_core.append(r"\end{tabular}")
    if wrap_resizebox:
        table_core.append(r"}")
    if caption:
        table_core.append(r"\caption{" + latex_escape(caption) + r"}")
    if label:
        table_core.append(r"\label{" + latex_escape(label) + r"}")
    table_core.append(r"\end{table*}")

    return "\n".join(table_core)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to input CSV file.")
    ap.add_argument("--out", required=True, help="Path to output .tex file.")
    ap.add_argument("--caption", default="Evaluation results on ImageNet-mini.",
                    help="LaTeX caption for the table.")
    ap.add_argument("--label", default="tab:results", help="LaTeX label for the table.")
    ap.add_argument("--no-resize", action="store_true",
                    help="Do not wrap the tabular in \\resizebox.")
    ap.add_argument("--datasets", nargs="+",
                    help="Only include rows with Dataset in this list (exact match, case-sensitive).")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, delimiter=",", quoting=csv.QUOTE_MINIMAL)

    # Optional dataset filtering
    if args.datasets:
        lower_map = {c.lower(): c for c in df.columns}
        if "dataset" not in lower_map:
            raise ValueError("Required column 'dataset' not found in CSV for filtering.")
        dataset_col = lower_map["dataset"]
        df = df[df[dataset_col].isin(set(args.datasets))].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("No rows remain after applying --datasets filter.")

    latex_code = generate_latex_table(
        df,
        caption=args.caption,
        label=args.label,
        wrap_resizebox=not args.no_resize,
    )

    Path(args.out).write_text(latex_code, encoding="utf-8")
    print(f"Wrote LaTeX table to: {args.out}")


if __name__ == "__main__":
    main()
