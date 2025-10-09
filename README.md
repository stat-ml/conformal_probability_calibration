## Conformal Probability Calibration — минимальный гайд

Короткий README для воспроизводимости в статье. Примеры запуска идут сразу.

### Быстрый старт (рекомендуется `uv`)

```bash
# 1) Python 3.13 и uv (https://astral.sh/uv). Создать/активировать окружение не обязательно —
#    можно просто использовать `uv run`.

# CIFAR (конфиг по умолчанию)
uv run -- python eval.py --config_file configs/config_cifar.json

# CIFAR с дополнительными флагами
uv run -- python eval.py --config_file configs/config_cifar.json \
  --cal-ratio 0.3 --num-splits 10 --subset-items 50000

# ImageNet-mini
uv run -- python eval.py --config_file configs/config_imagenet.json

# iNaturalist (пример с нестратегированной выборкой)
uv run -- python eval.py --config_file configs/config_inaturalist.json \
  --cal-ratio 0.3 --num-splits 1 --subset-items 30000 --do-not-stratify
```

### Что делает пайплайн
- Загружает датасет и модель, считает предсказания (кешируются в `test_preds.npz`).
- Делит данные на калибровку/тест по `--cal-ratio` (и числу сплитов).
- Применяет выбранные калибраторы и считает метрики.
- По желанию строит графики.

### Где править эксперименты
- Файлы конфигураций: `configs/config_cifar.json`, `configs/config_imagenet.json`, `configs/config_inaturalist.json`.
- Ключевые поля: `evaluations` (пары датасет–модель), `calibrators`, `metrics`, `visualizations`, `runner_settings`.
- Директория данных задаётся `data_root` (по умолчанию `data/`). Ожидается структура `data/<dataset_name>/...`.

### Выходные артефакты
- Папка берётся из `runner_settings.output_dir` (например, `experiments_cifar/`).
- Для каждой пары создаётся подкаталог `<dataset>_<model>/[split_i]/`.
- Сводные таблицы: `summary_results.txt` и/или `summary_results.csv` в корне `output_dir`.
- Кеш предсказаний: `<dataset>_<model>/test_preds.npz`.
- Графики (если включены в конфиг): сохраняются в соответствующие `split_i` директории.

### Зависимости
- Python 3.13 (`.python-version`).
- Проект использует `uv` и `pyproject.toml` для зависимостей. Запуск через `uv run` автоматически их подтянет.

### Каталогия проекта (минимально)
- `caliblab/` — датасеты, модели, калибраторы, метрики, визуализации и движок оценки.
- `eval.py` — единственная точка входа (CLI) с флагами:
  - `--config_file` (путь к JSON), `--num-splits`, `--cal-ratio`, `--subset-items`, `--do-not-stratify`.

Если нужен ещё более краткий TL;DR: запустите одну из команд из раздела «Быстрый старт».
