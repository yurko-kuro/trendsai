# trendsai/compare.py

import os
import math

from modules.alt.compare import (
    load_all_cluster_metrics,
    save_comparison_results,
)
from trendsai.cluster_modes import load_cluster_keywords
from modules.core.fetch import run as run_collect
from modules.report.pipeline import run as run_report


PREFERRED_METRICS_ORDER = ["smape", "mape", "mase", "rmse", "mae"]


def _is_valid_metric_value(v) -> bool:
    """Перевіряє, що метрика не None, не NaN, не ±inf."""
    if v is None:
        return False
    try:
        x = float(v)
    except (TypeError, ValueError):
        return False
    if math.isnan(x) or math.isinf(x):
        return False
    return True


def _auto_choose_metric(sample_row: dict) -> str:
    """
    Обирає найкращу метрику за замовчуванням із того, що реально є
    в *_metrics.json і має коректне значення.
    Пріоритет:
      smape -> mape -> mase -> rmse -> mae
    """
    for m in PREFERRED_METRICS_ORDER:
        if m in sample_row and _is_valid_metric_value(sample_row.get(m)):
            return m
    # якщо нічого хорошого не знайшли — fallback на rmse
    return "rmse"


def _ensure_metrics_for_keywords(
    keywords,
    db_path: str = "db.sqlite3",
    reports_dir: str = "reports",
    days: int = 1800,
    prediction_days: int = 30,
    seq_length: int = 12,
) -> None:
    """
    Гарантує, що для кожного слова є *_metrics.json.
    Якщо файлу немає — робить collect + report.
    """
    os.makedirs(reports_dir, exist_ok=True)

    for kw in keywords:
        base = kw.replace(" ", "_")
        metrics_path = os.path.join(reports_dir, f"{base}_metrics.json")
        if os.path.exists(metrics_path):
            continue

        print(f"[ENSURE] Немає метрик для '{kw}', запускаємо collect+report...")

        # collect: дані в БД
        run_collect(db_path=db_path, keywords=[kw], days=days)

        # report: тренування моделі + метрики
        code = run_report(
            kw,
            outdir=reports_dir,
            prediction_days=prediction_days,
            seq_length=seq_length,
        )
        if code != 0:
            print(f"[WARN] report для '{kw}' завершився з кодом {code}")


def compare_cluster(
    seed: str,
    metric: str,
    reports_dir: str,
    extra_keywords=None,
):
    """
    Порівнює всі слова у кластері seed-слова + додаткові слова:
      - читає список слів кластера з БД
      - додає extra_keywords (нові слова)
      - для всіх гарантує наявність *_metrics.json (collect+report при потребі)
      - автоматично обирає метрику (якщо metric='auto')
      - сортує та зберігає CSV/JSON
    """
    extra_keywords = extra_keywords or []

    # 1) базовий кластер з БД
    keywords = load_cluster_keywords("db.sqlite3", seed) or []

    # 2) додаємо нові слова з CLI
    for kw in extra_keywords:
        if kw not in keywords:
            keywords.append(kw)

    if not keywords:
        print("[ERR] Кластер порожній, немає що порівнювати.")
        return None

    # 3) гарантуємо, що є метрики для всіх слів
    _ensure_metrics_for_keywords(
        keywords,
        db_path="db.sqlite3",
        reports_dir=reports_dir,
        days=1800,
        prediction_days=30,
        seq_length=12,
    )

    # 4) вантажимо метрики (перший metric не критичний, при 'auto' все одно переоберемо)
    rows = load_all_cluster_metrics(
        keywords,
        metric if metric != "auto" else "rmse",
        reports_dir,
    )
    if not rows:
        print("[ERR] Не знайдено жодного *_metrics.json у кластері.")
        return None

    # 5) авто-вибір метрики з урахуванням NaN/inf/None
    if metric == "auto":
        metric = _auto_choose_metric(rows[0])
        print(
            f"[INFO] Обрана метрика за замовчуванням: {metric} "
            f"(пріоритет: {', '.join(PREFERRED_METRICS_ORDER)})"
        )

    # гарантуємо, що сортування враховує валідність значень
    def sort_key(row):
        v = row.get(metric)
        if not _is_valid_metric_value(v):
            return float("inf")
        return float(v)

    rows.sort(key=sort_key)

    print(f"\n=== Ранжування кластера '{seed}' за метрикою: {metric} ===")
    for i, row in enumerate(rows, start=1):
        kw = row["_keyword"]
        val = row.get(metric, None)
        print(f"{i:2d}. {kw:25s} {metric}={val}")

    best = rows[0]["_keyword"]

    res = save_comparison_results(rows, reports_dir, seed, metric)

    return {
        "ranked": rows,
        "best": best,
        "csv": res["csv"],
        "json": res["json"],
    }
