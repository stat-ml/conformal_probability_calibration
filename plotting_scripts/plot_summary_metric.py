import argparse
import re

import pandas as pd
import matplotlib.pyplot as plt


def parse_calibrator(value: str):
    """
    Возвращает (base_name, alpha) из строки Calibrator.
    Примеры:
      "cnfrml_mass_thrsh:a=0.01,sc.tp=aps,sc.trnf=iden" -> ("cnfrml_mass_thrsh", 0.01)
      "uncalibrated" -> ("uncalibrated", None)
    """
    if not isinstance(value, str):
        return None, None

    value = value.strip()
    if value.lower() == "uncalibrated":
        return "uncalibrated", None

    base = value.split(":", 1)[0].strip()
    m = re.search(r"a=([0-9.]+)", value)
    alpha = float(m.group(1)) if m else None
    return base, alpha


def parse_mean_std(cell):
    """
    Парсит ячейку вида '0.8148 ± 0.0010' -> (0.8148, 0.0010).
    Если число без '±', возвращает (value, 0.0).
    """
    if isinstance(cell, str):
        parts = re.split(r"\s*±\s*", cell)
        if len(parts) == 2:
            try:
                mean = float(parts[0])
                std = float(parts[1])
                return mean, std
            except ValueError:
                pass
    # fallback: просто число
    try:
        v = float(cell)
        return v, 0.0
    except (TypeError, ValueError):
        return float("nan"), float("nan")


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean ± std for a metric vs alpha for a given calibrator."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Путь к summary_results.csv",
    )
    parser.add_argument(
        "--calibrator",
        type=str,
        required=True,
        help="Базовое имя калибратора (например, 'cnfrml_mass_thrsh' или 'cnfrml_temp')",
    )
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help="Имя метрики (имя колонки в CSV, например 'nll', 'accuracy', 'coverage_[0.8, 0.82]', 'avg_set_size')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Если указано, сохранить график в файл (например, 'plot.png')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.dataset is not None:
        df = df[df["Dataset"] == args.dataset]

    if "Calibrator" not in df.columns:
        raise ValueError("В CSV нет колонки 'Calibrator'.")

    if args.metric not in df.columns:
        raise ValueError(
            f"В CSV нет метрики '{args.metric}'. Доступные колонки: {list(df.columns)}"
        )

    # Разбираем колонку Calibrator
    parsed = df["Calibrator"].apply(parse_calibrator)
    df["calib_base"] = parsed.apply(lambda x: x[0])
    df["alpha"] = parsed.apply(lambda x: x[1])

    # Фильтр по базовому калибратору и наличию alpha
    mask = (df["calib_base"] == args.calibrator) & df["alpha"].notna()
    sub = df[mask].copy()

    if sub.empty:
        raise ValueError(
            f"Не нашел строк для калибратора '{args.calibrator}' с параметром a=..."
        )

    # Парсим метрику в mean / std
    means_stds = sub[args.metric].apply(parse_mean_std)
    sub["metric_mean"] = means_stds.apply(lambda x: x[0])
    sub["metric_std"] = means_stds.apply(lambda x: x[1])

    sub = sub.sort_values("alpha")

    alphas = sub["alpha"].values
    means = sub["metric_mean"].values
    stds = sub["metric_std"].values

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        alphas,
        means,
        yerr=stds,
        fmt="-o",
        capsize=4,
        ecolor="black",
        elinewidth=1,
        markerfacecolor="blue",
    )
    plt.xlabel("Calibration alpha")
    plt.ylabel(args.metric)
    plt.title(f"{args.metric} vs alpha for {args.calibrator}")
    plt.grid(True)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()