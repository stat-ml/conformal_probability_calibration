import argparse
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_summary_metric import parse_calibrator, parse_mean_std


def find_suffix_coverage_columns(columns: List[str]) -> Dict[float, str]:
    """
    Находит все колонки вида 'alpha_suffix_coverage_<alpha>' и
    возвращает словарь {alpha: column_name}.
    """
    mapping: Dict[float, str] = {}
    pattern = re.compile(r"^alpha_suffix_coverage_([0-9.]+)$")
    for col in columns:
        m = pattern.match(col)
        if m:
            alpha_str = m.group(1)
            try:
                alpha = float(alpha_str)
            except ValueError:
                continue
            mapping[alpha] = col
    return mapping


def build_curve(
    df: pd.DataFrame,
    alpha_to_col: Dict[float, str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Строит зависимость:
      по оси X — alpha (из параметра калибратора a),
      по оси Y — значение в ячейке строки с тем же alpha, но в колонке alpha_suffix_coverage_<alpha>.

    То есть для каждого alpha находим строку с Calibrator-параметром a=alpha
    и колонку alpha_suffix_coverage_<alpha>, и берём значение метрики из этой ячейки.
    """
    if df["alpha"].isna().all():
        raise ValueError("Во всех строках alpha = NaN, нечего строить.")

    # alphas, которые присутствуют и как параметр калибратора, и как суффикс в названии колонки
    calib_alphas = sorted({a for a in df["alpha"].dropna().unique()})
    metric_alphas = sorted(alpha_to_col.keys())
    common_alphas = sorted(set(calib_alphas).intersection(metric_alphas))

    if not common_alphas:
        raise ValueError(
            "Не найдено ни одного alpha, который был бы и параметром калибратора, "
            "и суффиксом в колонке alpha_suffix_coverage_<alpha>."
        )

    xs: List[float] = []
    means: List[float] = []
    stds: List[float] = []

    for a in common_alphas:
        col = alpha_to_col[a]
        rows = df[df["alpha"] == a]
        if rows.empty:
            continue
        if len(rows) > 1:
            raise ValueError(
                f"Для alpha={a} найдено несколько строк после фильтрации. "
                "Уточните фильтры по Dataset/Model."
            )
        row = rows.iloc[0]
        cell = row[col]
        mean, std = parse_mean_std(cell)
        xs.append(a)
        means.append(mean)
        stds.append(std)

    if not xs:
        raise ValueError("Не удалось собрать ни одной точки для кривой.")

    return np.array(xs), np.array(means), np.array(stds)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Строит зависимость coverage_suffix_alpha от alpha для заданного калибратора.\n"
            "По оси X: alpha (параметр a калибратора в строке CSV).\n"
            "По оси Y: значение из ячейки той же строки, но в колонке alpha_suffix_coverage_<alpha>."
        )
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
        "--dataset",
        type=str,
        default=None,
        help="Если указано, фильтрация по названию датасета (колонка 'Dataset')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Если указано, фильтрация по названию модели (колонка 'Model')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Если указано, сохранить график в файл (например, 'plot_suffix_vs_alpha.png')",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Опциональная фильтрация по датасету и модели
    if args.dataset is not None:
        if "Dataset" not in df.columns:
            raise ValueError("В CSV нет колонки 'Dataset', а был указан аргумент --dataset.")
        df = df[df["Dataset"] == args.dataset]

    if args.model is not None:
        if "Model" not in df.columns:
            raise ValueError("В CSV нет колонки 'Model', а был указан аргумент --model.")
        df = df[df["Model"] == args.model]

    if df.empty:
        raise ValueError("После фильтрации по dataset/model в CSV не осталось строк.")

    if "Calibrator" not in df.columns:
        raise ValueError("В CSV нет колонки 'Calibrator'.")

    # Разбираем колонку Calibrator
    parsed = df["Calibrator"].apply(parse_calibrator)
    df["calib_base"] = parsed.apply(lambda x: x[0])
    df["alpha"] = parsed.apply(lambda x: x[1])

    # Фильтр по базовому калибратору
    df = df[df["calib_base"] == args.calibrator]
    if df.empty:
        raise ValueError(
            f"Не нашел строк для калибратора с базовым именем '{args.calibrator}'."
        )

    # Находим все alpha_suffix_coverage_* колонки
    alpha_to_col = find_suffix_coverage_columns(list(df.columns))
    if not alpha_to_col:
        raise ValueError(
            "Не найдено ни одной колонки 'alpha_suffix_coverage_<alpha>' в CSV."
        )

    xs_alpha, ys, ys_std = build_curve(df, alpha_to_col)
    # По оси X хотим 1 - alpha
    xs = 1.0 - xs_alpha

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        xs,
        ys,
        yerr=ys_std,
        fmt="-o",
        capsize=4,
        elinewidth=1,
        markerfacecolor="blue",
    )

    plt.xlabel("1 - alpha (параметр a калибратора)")
    plt.ylabel("alpha_suffix_coverage_alpha")
    title = f"coverage_suffix_alpha vs (1 - alpha) for {args.calibrator}"
    if args.dataset:
        title += f" on {args.dataset}"
    if args.model:
        title += f" ({args.model})"
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()


