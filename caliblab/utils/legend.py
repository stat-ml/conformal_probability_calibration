from __future__ import annotations

from typing import Optional



def map_legend_label(raw_name: Optional[str]) -> str:
    """Normalize calibrator names for plot legends.

    Current rules:
    - Uncalibrated/Uncaibrated/none -> "Base"
    - Keep other names as-is (after normalizing Cyrillic "с/С" to Latin "c/C").

    Args:
        raw_name: Original calibrator name (can be None).

    Returns:
        A display-friendly legend label.
    """
    if raw_name is None:
        return ""

    base = raw_name.split(":", 1)[0].strip().lower()

    if base in {"uncalibrated", "uncaibrated", "none"}:
        return "Base"
    
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
    if base.startswith("cnfrml_mass_thrsh"):
        return "MR (ours)"
    if base.startswith("cnfrml_temp"):
        return "TS (ours)"
    if base.startswith("cnfrml_"):
        suffix = base[len("cnfrml_"):].replace("_", " ")
        return f"ours({suffix})"

    return base


__all__ = ["map_legend_label"]


