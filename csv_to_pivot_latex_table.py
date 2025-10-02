#!/usr/bin/env python3
r"""
Create a LaTeX pivot table from a CSV.

Features:
- Dataset and Model filters (exact match, case-sensitive)
- Choose metric column to display in each cell
- Choose row and column index fields via CLI
- Caption above the table; label optional
- Uses booktabs; values printed as-is (no ± parsing here)

Example:
  uv run -- python csv_to_pivot_latex_table.py \
    --csv experiments_cifar/summary_results.csv \
    --out experiments_cifar/summary_results_pivot.tex \
    --rows Calibrator \
    --cols Dataset \
    --metric ece \
    --datasets cifar100 \
    --models ResNet56
"""

import argparse
import csv
import difflib as _difflib
from pathlib import Path
from typing import List, Dict

import pandas as pd


def latex_escape(text: str) -> str:
    if text is None:
        return ""
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

# New: escape without touching backslashes (to keep LaTeX commands like \pm)
def latex_escape_keep_cmd(text: str) -> str:
    if text is None:
        return ""
    replacements = {
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


# --- Helpers to derive fields from Calibrator ---
Cyrillic_C_MAP = str.maketrans({"с": "c", "С": "C"})

def parse_calibrator_meta(name: str) -> Dict[str, str]:
    meta = {"alpha": "", "score_type": "", "transform": ""}
    if name is None:
        return meta
    s = str(name).translate(Cyrillic_C_MAP)
    if ":" not in s:
        return meta
    try:
        _, params = s.split(":", 1)
    except ValueError:
        return meta
    for part in params.split(","):
        kv = part.strip()
        if "=" not in kv:
            continue
        key, val = kv.split("=", 1)
        key = key.strip().lower()
        val = val.strip()
        if key == "a":
            meta["alpha"] = val
        elif key in {"sc.tp", "sc_type", "score_type", "sctp"}:
            meta["score_type"] = val
        elif key in {"sc.trnf", "sc.tr", "transform", "sctrnf"}:
            meta["transform"] = val
    return meta

def map_transform_display(value: str) -> str:
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

def map_calibrator_type(name: str) -> str:
    if name is None:
        return ""
    s = str(name).translate(Cyrillic_C_MAP)
    base = s.split(":", 1)[0].strip().lower()
    if base in {"uncalibrated", "none"}:
        return "Base"
    if base.startswith("cnfrml_mass_thrsh"):
        return "CMT (ours)"
    if base.startswith("cnfrml_temp"):
        return "TS (ours)"
    # Any other calibrator → "other" for pivot grouping
    return "other"

def map_calibrator_display(name: str) -> str:
    """Pretty display for calibrator values in pivot rows/columns."""
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
    if base.startswith("cnfrml_mass_thrsh"):
        return "CMT(ours)"
    if base.startswith("cnfrml_temp"):
        return "TS(ours)"
    return s

def map_score_type_display(value: str) -> str:
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
    if name is None:
        return ""
    key = str(name).strip().lower()
    if key == "alpha":
        return "<<ALPHA>>"
    if key == "score_type":
        return "Score"
    if key == "transform":
        return "Transf."
    return str(name)

def _map_value_for_field(field_name: str, value: object) -> str:
    fname = (field_name or "").strip().lower()
    if fname == "calibrator":
        return map_calibrator_display(value)
    if fname == "score_type":
        return map_score_type_display(value)
    if fname == "transform":
        return map_transform_display(value)
    return str(value)

def _resolve_fields(user_fields: List[str], available_columns: List[str]) -> List[str]:
    """Resolve user-provided field names to actual DataFrame columns.
    - Case-insensitive match first
    - Fallback to close match using difflib
    """
    lower_to_actual = {c.lower(): c for c in available_columns}
    resolved: List[str] = []
    for name in user_fields:
        key = str(name).strip()
        if key in available_columns:
            resolved.append(key)
            continue
        lk = key.lower()
        if lk in lower_to_actual:
            resolved.append(lower_to_actual[lk])
            continue
        # Try fuzzy match among lowercased available
        candidates = list(lower_to_actual.keys())
        best = _difflib.get_close_matches(lk, candidates, n=1, cutoff=0.8)
        if best:
            resolved.append(lower_to_actual[best[0]])
            continue
        raise ValueError(f"Unknown column '{name}'. Available: {available_columns}")
    return resolved


def build_pivot_table(df: pd.DataFrame, row_fields: List[str], col_fields: List[str], metric: str) -> pd.DataFrame:
    # Pivot and fill missing with ""
    if not row_fields:
        raise ValueError("--rows must have at least one column name")
    if not col_fields:
        raise ValueError("--cols must have at least one column name")
    if metric not in df.columns:
        raise ValueError(f"Metric column '{metric}' not found in CSV")
    pivot = pd.pivot_table(df, index=row_fields, columns=col_fields, values=metric, aggfunc='first')
    return pivot.fillna("")


def pivot_to_latex(pivot: pd.DataFrame, caption: str, label: str, wrap_resizebox: bool = True) -> str:
    # helper to parse numeric part for min comparisons
    def _to_number(cell: object):
        if cell is None:
            return None
        txt = str(cell).strip()
        if txt in {"", "--"}:
            return None
        for sep in ["±", "+/-"]:
            if sep in txt:
                txt = txt.split(sep, 1)[0].strip()
                break
        try:
            return float(txt)
        except Exception:
            return None

    # left (row index) headers
    row_index_names = [map_header_display(name if name is not None else "") for name in pivot.index.names]
    num_left = len(row_index_names)

    # right (column) headers with two-line support
    is_multi_cols = isinstance(pivot.columns, pd.MultiIndex) and pivot.columns.nlevels >= 2
    col_level_names = list(pivot.columns.names) if hasattr(pivot.columns, 'names') else []
    if is_multi_cols:
        leaf_cols = list(pivot.columns)
        top_labels = [_map_value_for_field(col_level_names[0] if col_level_names else "", t[0]) for t in leaf_cols]
        bottom_parts = []
        for t in leaf_cols:
            parts = []
            for lvl, v in enumerate(t[1:], start=1):
                fname = col_level_names[lvl] if lvl < len(col_level_names) else ""
                parts.append(_map_value_for_field(fname, v))
            bottom_parts.append(" ".join(parts) if parts else "")
        bottom_labels = bottom_parts
        # compute multicolumn spans for identical consecutive top labels
        spans = []
        last = None
        span = 0
        for lbl in top_labels:
            if lbl == last:
                span += 1
            else:
                if last is not None:
                    spans.append((last, span))
                last, span = lbl, 1
        if last is not None:
            spans.append((last, span))
        col_count = len(leaf_cols)
    else:
        leaf_cols = list(pivot.columns)
        first_level_name = col_level_names[0] if col_level_names else ""
        col_headers = [_map_value_for_field(first_level_name, c) for c in leaf_cols]
        col_count = len(col_headers)

    alignment = " ".join(["l"] * num_left + ["c"] * col_count)

    # compute minima for rows and columns
    numeric_df = pivot.applymap(_to_number)
    row_mins = numeric_df.min(axis=1, skipna=True)
    col_mins = numeric_df.min(axis=0, skipna=True)

    lines: List[str] = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    if caption:
        lines.append(r"\caption{" + latex_escape(caption) + r"}")
    if label:
        lines.append(r"\label{" + latex_escape(label) + r"}")
    if wrap_resizebox:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + alignment + r"}")
    lines.append(r"\toprule")

    # header line 1
    # Apply alpha placeholder after escaping later
    h1_left = [latex_escape_keep_cmd(row_index_names[0] if num_left >= 1 else "")] + [""] * (num_left - 1)
    header_row_1 = h1_left
    if is_multi_cols:
        for lbl, span in spans:
            header_row_1.append(r"\multicolumn{" + str(span) + r"}{c}{" + latex_escape_keep_cmd(lbl) + r"}")
    else:
        header_row_1 += [latex_escape_keep_cmd(h) for h in col_headers]
    line1 = " & ".join(header_row_1) + r" \\"
    line1 = line1.replace("<<ALPHA>>", r"$\alpha$")
    lines.append(line1)

    # add column separators under top-level headers
    if is_multi_cols:
        start_col = num_left + 1  # 1-based indexing in LaTeX tabular
        cur = start_col
        for _lbl, span in spans:
            a = cur
            b = cur + span - 1
            lines.append(r"\cmidrule(lr){" + str(a) + "-" + str(b) + r"}")
            cur = b + 1

    # header line 2
    h2_left = [latex_escape_keep_cmd(row_index_names[1] if num_left >= 2 else "")] + [""] * (num_left - 1)
    header_row_2 = h2_left
    if is_multi_cols:
        header_row_2 += [latex_escape_keep_cmd(h) for h in bottom_labels]
    else:
        header_row_2 += [""] * col_count
    line2 = " & ".join(header_row_2) + r" \\"
    line2 = line2.replace("<<ALPHA>>", r"$\\alpha$")
    lines.append(line2)
    lines.append(r"\midrule")

    # body rows with duplicate index suppression and min highlighting
    prev_idx = None
    eps = 1e-12
    for ridx, (idx_vals, row) in enumerate(pivot.iterrows()):
        if not isinstance(idx_vals, tuple):
            idx_vals = (idx_vals,)
        # group separator when top-level index changes
        if ridx > 0 and len(idx_vals) >= 1 and prev_idx is not None and prev_idx[0] != idx_vals[0]:
            lines.append(r"\addlinespace")
        left_cells = []
        for level, val in enumerate(idx_vals):
            field_name = pivot.index.names[level] if level < len(pivot.index.names) else ""
            display = _map_value_for_field(field_name, val)
            if prev_idx is not None and level < len(prev_idx):
                if all(prev_idx[k] == idx_vals[k] for k in range(level)) and prev_idx[level] == idx_vals[level]:
                    display = ""
            cell = latex_escape_keep_cmd(display)
            cell = cell.replace("<<ALPHA>>", r"$\alpha$")
            left_cells.append(cell)
        prev_idx = idx_vals

        right_cells = []
        for cidx, val in enumerate(row.tolist()):
            # Replace ASCII '+/-' and Unicode '±' with LaTeX \pm, then escape (keeping commands)
            txt = str(val).replace("+/-", r"\pm").replace("±", r"\pm")
            esc = latex_escape_keep_cmd(txt)
            vnum = _to_number(val)
            bold = False
            underline = False
            if vnum is not None:
                rmin = row_mins.iloc[ridx]
                cmin = col_mins.iloc[cidx]
                if pd.notna(rmin) and abs(vnum - float(rmin)) <= eps:
                    bold = True
                if pd.notna(cmin) and abs(vnum - float(cmin)) <= eps:
                    underline = True
            if bold and underline:
                esc = r"\underline{\mathbf{" + esc + r"}}"
            elif bold:
                esc = r"\mathbf{" + esc + r"}"
            elif underline:
                esc = r"\underline{" + esc + r"}"
            # wrap in math mode if non-empty
            if esc.strip() != "":
                esc = "$" + esc + "$"
            right_cells.append(esc)

        lines.append(" & ".join(left_cells + right_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if wrap_resizebox:
        lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to input CSV file.")
    ap.add_argument("--out", required=True, help="Path to output .tex file.")
    ap.add_argument("--rows", nargs="+", required=True, help="Columns to use as pivot table rows (ordered).")
    ap.add_argument("--cols", nargs="+", required=True, help="Columns to use as pivot table columns (ordered).")
    ap.add_argument("--metric", required=True, help="Metric column name to display in each cell.")
    ap.add_argument("--caption", default="Pivot results.", help="LaTeX caption for the table.")
    ap.add_argument("--label", default="tab:pivot", help="LaTeX label for the table.")
    ap.add_argument("--no-resize", action="store_true", help="Do not wrap the tabular in \\resizebox.")
    ap.add_argument("--datasets", nargs="+", help="Filter rows where Dataset is in this list (exact match).")
    ap.add_argument("--models", nargs="+", help="Filter rows where Model is in this list (exact match).")
    ap.add_argument("--calib-pattern", default=None,
                    help="Regex to include only calibrators matching the pattern (case-insensitive). Applied after dataset/model filters.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, delimiter=",", quoting=csv.QUOTE_MINIMAL)

    # Optional dataset/model filtering
    lower_map = {c.lower(): c for c in df.columns}
    if args.datasets:
        if "dataset" not in lower_map:
            raise ValueError("Required column 'Dataset' not found for filtering.")
        df = df[df[lower_map["dataset"]].isin(set(args.datasets))].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("No rows remain after --datasets filter.")
    if args.models:
        if "model" not in lower_map:
            raise ValueError("Required column 'Model' not found for filtering.")
        df = df[df[lower_map["model"]].isin(set(args.models))].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("No rows remain after --models filter.")

    # Optional calibrator regex include filter
    if args.calib_pattern:
        if "calibrator" not in lower_map:
            raise ValueError("Required column 'Calibrator' not found for filtering by --calib-pattern.")
        import re as _re
        try:
            rx = _re.compile(args.calib_pattern, _re.IGNORECASE)
        except _re.error as e:
            raise ValueError(f"Invalid --calib-pattern regex: {e}")
        cal_col = lower_map["calibrator"]
        df = df[df[cal_col].astype(str).apply(lambda s: bool(rx.search(s)))].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("No rows remain after --calib-pattern filter.")

    # Derive helper columns if present: alpha, score_type, transform, calibrator_type
    lower_map = {c.lower(): c for c in df.columns}
    if "calibrator" in lower_map:
        cal_col = lower_map["calibrator"]
        metas = df[cal_col].apply(parse_calibrator_meta)
        df["alpha"] = metas.apply(lambda m: m.get("alpha", ""))
        df["score_type"] = metas.apply(lambda m: m.get("score_type", ""))
        df["transform"] = metas.apply(lambda m: map_transform_display(m.get("transform", "")))
        df["calibrator_type"] = df[cal_col].apply(map_calibrator_type)

    # Resolve user fields robustly (case-insensitive, fuzzy)
    row_fields = _resolve_fields(args.rows, list(df.columns))
    col_fields = _resolve_fields(args.cols, list(df.columns))

    pivot = build_pivot_table(df, row_fields, col_fields, args.metric)
    latex = pivot_to_latex(pivot, args.caption, args.label, wrap_resizebox=not args.no_resize)
    Path(args.out).write_text(latex, encoding="utf-8")
    print(f"Wrote LaTeX pivot table to: {args.out}")


if __name__ == "__main__":
    main()


