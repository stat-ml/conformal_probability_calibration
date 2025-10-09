#!/usr/bin/env python3
r"""
Generate a LaTeX table from a CSV with metrics like "mean ± std".

Rules:
- Best/second-best highlighting per column (entire formula, not just mean):
    * Best: bold
    * Second-best: underline
    * accuracy: maximize (highest mean is best)
    * nll, brier_score, ece, mce, cw-ece, cmce: minimize (lowest mean is best)
- Coverage columns named like: coverage_[L, U]
    * Bold and underline if the mean is within [L, U] (inclusive), regardless of ranking.
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
    # Normalize Cyrillic 'с'/'С' to Latin 'c'/'C' to avoid visually similar characters
    out = str(text)
    try:
        out = out.translate(Cyrillic_C_MAP)
    except NameError:
        # Map may not be defined yet at import time; skip gracefully
        out = out
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
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out

Cyrillic_C_MAP = str.maketrans({"с": "c", "С": "C"})

def parse_calibrator_meta(name: str) -> Dict[str, str]:
    """Extract alpha (a), score_type (sc.tp), and transform (sc.trnf) from calibrator string.
    Returns empty strings when not present.
    Handles Cyrillic 'с' in keys by normalizing to Latin 'c'.
    """
    result = {"alpha": "", "score_type": "", "transform": ""}
    if not name:
        return result
    s = str(name)
    # Normalize potential Cyrillic characters in the entire string
    s = s.translate(Cyrillic_C_MAP)
    if ":" not in s:
        return result
    try:
        _prefix, params = s.split(":", 1)
    except ValueError:
        return result
    for part in params.split(","):
        kv = part.strip()
        if "=" not in kv:
            continue
        key, val = kv.split("=", 1)
        key = key.strip().lower()
        val = val.strip()
        if key == "a":
            result["alpha"] = val
        elif key in {"sc.tp", "sc_type", "score_type", "sctp"}:
            result["score_type"] = val
        elif key in {"sc.trnf", "sc.tr", "transform", "sctrnf"}:
            result["transform"] = val
    return result

def map_calibrator_display(name: str) -> str:
    """Map raw calibrator names to display labels.
    - cnfrml_mass_thrsh:* -> ours(cumulative_mass_threshold)
    - cnfrml_temp:*       -> ours(temperature_scaling)
    - cnfrml_*            -> ours(<suffix>)
    Other names shown as-is. Normalizes Cyrillic 'с'/'С'.
    """
    if name is None:
        return ""
    s = str(name).translate(Cyrillic_C_MAP)
    base = s.split(":", 1)[0].strip().lower()
    if base in {"uncalibrated", "none"}:
        return "Base"
    if base.startswith("isotonic"):
        return "Isotonic"
    if base == "venn_abers_one_vs_all":
        return "V.-Abers (OvA)"
    if base in {"temp_scaling", "temperature_scaling"}:
        return "Temp. scaling"
    if base == "platt_regression":
        return "Platt Scaling"
    if base == "dirichlet":
        return "Dirichlet"
    if base in {"adaptive_temperature_scaling", "ada_temp_scaling"}:
        return "Ada-Temp Scaling"
    # Normalize Naive CMCE calibrator naming
    if base in {"naive cmce calibrator", "naive_cmce_calibrator", "naive cmce", "naive_cmce"}:
        return "Naive CMCE"
    if base.startswith("cnfrml_mass_thrsh"):
        return "MR (ours)"
    if base.startswith("cnfrml_temp"):
        return "TS (ours)"
    if base.startswith("cnfrml_"):
        suffix = base[len("cnfrml_"):].replace("_", " ")
        return f"ours({suffix})"
    return s

def map_transform_display(value: str) -> str:
    """Map transform short codes to display names."""
    if value is None:
        return ""
    s = str(value).strip().lower()
    if s in {"temp_scaling", "temperature_scaling", "temp"}:
        return "Temp. scaling"
    if s in {"norm", "norm_flow"}:
        return "NF"
    if s in {"iden", "identity"}:
        return "I"
    return str(value)

def map_score_type_display(value: str) -> str:
    """Map score_type values to display names."""
    if value is None:
        return ""
    s = str(value).strip().lower()
    if s == "aps":
        return "APS"
    if s == "thr":
        return "MSP"
    if s == "temp_scaling":
        return "Temp. scaling"
    return str(value)

def map_header_display(name: str) -> str:
    """Map header column display names, including text and metric columns.
    Uses a placeholder token for math to avoid escaping (replaced after escaping step).
    """
    if name is None:
        return ""
    key = str(name).strip().lower()
    if key == "alpha":
        return "<<ALPHA>>"
    if key == "score_type":
        return "Score"
    if key == "transform":
        return "Transf."
    return map_metric_header_display(name)

def map_metric_header_display(name: str) -> str:
    """Pretty-print metric header names for LaTeX output.
    Only adjusts display casing for specific metrics like ECE/MCE.
    """
    if name is None:
        return ""
    s = str(name)
    key = s.strip().lower()
    # Coverage columns -> "Coverage [L, U]"
    cov = is_coverage_col(s)
    if cov is not None:
        lo, hi = cov
        return f"Coverage [{lo}, {hi}]"
    if key == "cw-ece":
        return "cw-ECE"
    if key == "nll":
        return "NLL"
    if key == "ece":
        return "ECE"
    if key == "mce":
        return "MCE"
    if key == "cmce":
        return "CMCE"
    if key == "brier_score":
        return "Brier Score"
    return s

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

def find_top2_indices(values: List[Optional[Tuple[float, float]]],
                      minimize: bool,
                      eps: float = 1e-12) -> Tuple[List[int], List[int]]:
    """Return (best_indices, second_best_indices) based on means.

    - Includes ties in each rank set.
    - Ignores rows with missing values (treated as +inf/-inf already).
    - If there is no distinct second value, second_best_indices is empty.
    """
    means = [v[0] if v else (math.inf if minimize else -math.inf) for v in values]
    # Filter out placeholder infinities that represent non-numeric cells
    finite_pairs: List[Tuple[float, int]] = []
    for idx, m in enumerate(means):
        if minimize and m == math.inf:
            continue
        if not minimize and m == -math.inf:
            continue
        finite_pairs.append((m, idx))

    if not finite_pairs:
        return [], []

    # Determine best value and indices
    if minimize:
        best_val = min(m for m, _ in finite_pairs)
    else:
        best_val = max(m for m, _ in finite_pairs)
    best_indices = [i for m, i in finite_pairs if abs(m - best_val) <= eps]

    # Candidates for second-best must be distinct from best
    candidate_vals = [m for m, _ in finite_pairs if abs(m - best_val) > eps]
    if not candidate_vals:
        return best_indices, []
    if minimize:
        second_val = min(candidate_vals)
    else:
        second_val = max(candidate_vals)
    second_indices = [i for m, i in finite_pairs if abs(m - second_val) <= eps]
    return best_indices, second_indices

def format_math(mean: float, std: float, bold: bool = False, underline: bool = False) -> str:
    """Format a number as LaTeX math with optional bold and/or underline for the whole formula."""
    content = f"{mean:.4f} \\pm {std:.4f}"
    if bold:
        content = r"\mathbf{" + content + r"}"
    if underline:
        content = r"\underline{" + content + r"}"
    return r"$" + content + r"$"

# -------- Main generation --------

def generate_latex_table(df: pd.DataFrame,
                         caption: str,
                         label: str,
                         wrap_resizebox: bool = True,
                         context_df: Optional[pd.DataFrame] = None) -> str:
    # Identify available text columns and metric columns
    # Prefer these text columns if present (order matters for display):
    # Place calibrator-derived fields immediately after Calibrator
    preferred_text_cols = ["dataset", "model", "calibrator", "alpha", "score_type", "transform"]
    lower_map = {c.lower(): c for c in df.columns}
    text_cols = [lower_map[c] for c in preferred_text_cols if c in lower_map]
    dataset_col = lower_map.get("dataset")
    model_col = lower_map.get("model")
    calibrator_col = lower_map.get("calibrator")

    # Context columns (for sorting and baseline coloring) default to df if not provided
    ctx = context_df if context_df is not None else df
    lower_map_ctx = {c.lower(): c for c in ctx.columns}
    dataset_col_ctx = lower_map_ctx.get("dataset")
    model_col_ctx = lower_map_ctx.get("model")
    calibrator_col_ctx = lower_map_ctx.get("calibrator")
    metric_cols = [c for c in df.columns if c not in set(text_cols)]

    # Pre-parse all metric cells into (mean, std) or None for non-numeric
    parsed: Dict[str, List[Optional[Tuple[float, float]]]] = {}
    for col in metric_cols:
        parsed[col] = [parse_pm(x) for x in df[col].tolist()]

    # Determine best and second-best rows for each metric (non-coverage)
    top2_indices_by_col: Dict[str, Tuple[List[int], List[int]]] = {}
    for col in metric_cols:
        if is_coverage_col(col):
            continue  # handled via interval rule
        vals = parsed[col]
        if is_higher_better(col):
            best_idxs, second_idxs = find_top2_indices(vals, minimize=False)
        elif is_lower_better(col):
            best_idxs, second_idxs = find_top2_indices(vals, minimize=True)
        else:
            # If unknown metric, default to minimizing (conservative)
            best_idxs, second_idxs = find_top2_indices(vals, minimize=True)
        top2_indices_by_col[col] = (best_idxs, second_idxs)

    # Build LaTeX header
    header_cols = text_cols + metric_cols
    header_latex = " & ".join(latex_escape(map_header_display(c)) for c in header_cols) + r" \\" 
    # Replace math placeholders post-escape to keep LaTeX intact
    header_latex = header_latex.replace("<<ALPHA>>", r"$\alpha$")

    # Column alignment matches available text columns
    alignment = " ".join(["l"] * len(text_cols) + ["c"] * len(metric_cols))

    # Pre-compute uncalibrated baselines per (Dataset, Model) when possible
    def _is_uncalibrated(name: str) -> bool:
        s = str(name).strip().lower()
        return s in {"uncalibrated", "none"}

    def _group_key(i: int) -> tuple:
        if dataset_col_ctx is not None and model_col_ctx is not None:
            return (str(ctx.at[i, dataset_col_ctx]), str(ctx.at[i, model_col_ctx]))
        # Fallback: single key per entire table
        return ("__all__",)

    baseline_index_by_group: Dict[tuple, int] = {}
    if dataset_col_ctx is not None and model_col_ctx is not None and calibrator_col_ctx is not None:
        for i in range(len(ctx)):
            if _is_uncalibrated(ctx.at[i, calibrator_col_ctx]):
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
    if calibrator_col_ctx is not None:
        ordered_indices = sorted(indices, key=lambda i: (_calibrator_group(ctx.at[i, calibrator_col_ctx]), i))
    else:
        ordered_indices = indices

    body_lines: List[str] = []
    prev_group: Optional[int] = None
    for i in ordered_indices:
        if calibrator_col_ctx is not None:
            cur_group = _calibrator_group(ctx.at[i, calibrator_col_ctx])
            if prev_group is not None and cur_group > prev_group:
                body_lines.append(r"\\")
            prev_group = cur_group
        row_cells: List[str] = []
        # text cells
        for col in text_cols:
            if calibrator_col is not None and col == calibrator_col:
                row_cells.append(latex_escape(map_calibrator_display(df.at[i, col])))
            elif col.lower() == "transform":
                row_cells.append(latex_escape(map_transform_display(df.at[i, col])))
            elif col.lower() == "score_type":
                row_cells.append(latex_escape(map_score_type_display(df.at[i, col])))
            else:
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
            underline = False
            if cov_bounds is not None:
                lo, hi = cov_bounds
                # Bold+underline if mean in [lo, hi]
                if mean >= lo - 1e-12 and mean <= hi + 1e-12:
                    bold, underline = True, True
            else:
                # Bold if best, underline if second-best
                best_idxs, second_idxs = top2_indices_by_col.get(col, ([], []))
                if i in best_idxs:
                    bold = True
                elif i in second_idxs:
                    underline = True

            cell = format_math(mean, std, bold=bold, underline=underline)

            # Optional green/red color if better/worse than uncalibrated baseline (non-coverage metrics)
            if dataset_col_ctx is not None and model_col_ctx is not None and calibrator_col_ctx is not None:
                gk = _group_key(i)
                base_idx = baseline_index_by_group.get(gk)
                if base_idx is not None and cov_bounds is None:
                    base_val = parsed[col][base_idx]
                    if base_val is not None:
                        base_mean, _ = base_val
                        if _is_better(col, mean, base_mean):
                            cell = r"\textcolor{ForestGreen}{" + cell + r"}"
                        elif _is_worse(col, mean, base_mean):
                            cell = r"\textcolor{red}{" + cell + r"}"

            row_cells.append(cell)

        body_lines.append(" & ".join(row_cells) + r" \\")
    # Compose full table with booktabs
    table_core = []
    table_core.append(r"\begin{table*}[htbp]")
    table_core.append(r"\centering")
    if caption:
        table_core.append(r"\caption{" + latex_escape(caption) + r"}")
    if label:
        table_core.append(r"\label{" + latex_escape(label) + r"}")
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
    ap.add_argument("--models", nargs="+",
                    help="Only include rows with Model in this list (exact match, case-sensitive).")
    ap.add_argument("--exclude-cols", nargs="+",
                    help="Column names to exclude (case-insensitive). Accepts coverage_[l,u] regardless of spaces.")
    ap.add_argument("--calib-pattern", default=None,
                    help="Regex to include only calibrators matching the pattern (case-insensitive). Applied after dataset/model filters.")
    ap.add_argument("--calib-exclude-pattern", default=None,
                    help="Regex to exclude calibrators matching the pattern (case-insensitive). Applied after --calib-pattern include filter.")

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

    # Optional model filtering
    if getattr(args, "models", None):
        lower_map = {c.lower(): c for c in df.columns}
        if "model" not in lower_map:
            raise ValueError("Required column 'model' not found in CSV for filtering by --models.")
        model_col = lower_map["model"]
        df = df[df[model_col].isin(set(args.models))].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("No rows remain after applying --models filter.")

    # Optional calibrator regex include filter
    if getattr(args, "calib_pattern", None):
        lower_map = {c.lower(): c for c in df.columns}
        if "calibrator" not in lower_map:
            raise ValueError("Required column 'calibrator' not found in CSV for filtering by --calib-pattern.")
        cal_col = lower_map["calibrator"]
        try:
            rx = re.compile(args.calib_pattern, re.IGNORECASE)
        except re.error as e:
            raise ValueError(f"Invalid --calib-pattern regex: {e}")
        df = df[df[cal_col].astype(str).apply(lambda s: bool(rx.search(s)))].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("No rows remain after applying --calib-pattern filter.")

    # Optional calibrator regex exclude filter
    if getattr(args, "calib_exclude_pattern", None):
        lower_map = {c.lower(): c for c in df.columns}
        if "calibrator" not in lower_map:
            raise ValueError("Required column 'calibrator' not found in CSV for filtering by --calib-exclude-pattern.")
        cal_col = lower_map["calibrator"]
        try:
            rx_ex = re.compile(args.calib_exclude_pattern, re.IGNORECASE)
        except re.error as e:
            raise ValueError(f"Invalid --calib-exclude-pattern regex: {e}")
        df = df[~df[cal_col].astype(str).apply(lambda s: bool(rx_ex.search(s)))].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("No rows remain after applying --calib-exclude-pattern filter.")

    # Preserve a context copy (after row-filters, before column exclusions) for coloring/sorting
    context_df = df.copy()
    # Derive calibrator meta columns from Calibrator, if present
    lower_map = {c.lower(): c for c in df.columns}
    if "calibrator" in lower_map:
        cal_col = lower_map["calibrator"]
        metas = df[cal_col].apply(parse_calibrator_meta)
        df["alpha"] = metas.apply(lambda m: m.get("alpha", ""))
        df["score_type"] = metas.apply(lambda m: m.get("score_type", ""))
        df["transform"] = metas.apply(lambda m: m.get("transform", ""))

    # Optional column exclusion
    if getattr(args, "exclude_cols", None):
        lower_map = {c.lower(): c for c in df.columns}
        # Normalize coverage names in both CSV and requested excludes by stripping spaces after comma
        def normalize_cov(name: str) -> str:
            m = is_coverage_col(name) if name is not None else None
            if m is None:
                return str(name).strip().lower()
            lo, hi = m
            return f"coverage_[{lo}, {hi}]".lower()

        norm_csv_map: Dict[str, str] = {}
        for c in df.columns:
            norm_csv_map[normalize_cov(c)] = c

        drop_cols = []
        for name in args.exclude_cols:
            norm = normalize_cov(str(name))
            # Try exact lower-case name first
            candidate = lower_map.get(norm)
            if candidate is None:
                # Try normalized coverage mapping
                candidate = norm_csv_map.get(norm)
            if candidate is not None:
                drop_cols.append(candidate)
        if drop_cols:
            drop_cols = list(dict.fromkeys(drop_cols))
            df = df.drop(columns=drop_cols)

    latex_code = generate_latex_table(
        df,
        caption=args.caption,
        label=args.label,
        wrap_resizebox=not args.no_resize,
        context_df=context_df,
    )

    Path(args.out).write_text(latex_code, encoding="utf-8")
    print(f"Wrote LaTeX table to: {args.out}")


if __name__ == "__main__":
    main()
