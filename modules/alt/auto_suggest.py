# modules/alt/auto_suggest.py

import os
import json
import time
from typing import List, Dict

from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
from prettytable import PrettyTable
from requests.exceptions import RequestException


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -------------------------------------------------
# Google Trends: related queries
# -------------------------------------------------

def get_related_keywords(
    base_keyword: str,
    geo: str = "UA",
    hl: str = "uk-UA",
    top_n: int = 10,
    attempts: int = 5,
    base_sleep: int = 60,
) -> List[str]:
    """Отримання related-запитів з Google Trends з ретраями."""
    pytrends = TrendReq(hl=hl, tz=0)

    last_error = None
    related = None

    for attempt in range(1, attempts + 1):
        try:
            pytrends.build_payload([base_keyword], timeframe="today 5-y", geo=geo)
            related = pytrends.related_queries()
            break

        except TooManyRequestsError as e:
            last_error = e
            sleep_for = base_sleep * (2 ** (attempt - 1))
            print(
                f"[TRENDS] 429 TooManyRequests для '{base_keyword}', "
                f"спроба {attempt}/{attempts}, чекаємо ~{sleep_for} c..."
            )
            time.sleep(sleep_for)

        except Exception as e:
            last_error = e
            print(f"[TRENDS] Помилка для '{base_keyword}': {e}")
            break

    if last_error and related is None:
        print(
            f"[TRENDS] Не вдалося отримати related-запити для '{base_keyword}' "
            f"після {attempts} спроб. Помилка: {last_error}"
        )
        return []

    if not related or base_keyword not in related:
        return []

    res = related[base_keyword]
    candidates = []

    top_df = res.get("top")
    if top_df is not None:
        candidates.extend(top_df["query"].tolist())

    rising_df = res.get("rising")
    if rising_df is not None:
        candidates.extend(rising_df["query"].tolist())

    uniq = []
    seen = set()
    for q in candidates:
        q = str(q).strip()
        if not q or q.lower() == base_keyword.lower():
            continue
        if q not in seen:
            seen.add(q)
            uniq.append(q)

    return uniq[:top_n]


# -------------------------------------------------
# Backward compatibility
# -------------------------------------------------

def discover_candidates_via_trends(
    base_keyword: str,
    geo: str = "UA",
    hl: str = "uk-UA",
    top_n: int = 10,
) -> List[str]:
    """Сумісність зі старим cluster_modes."""
    return get_related_keywords(
        base_keyword=base_keyword,
        geo=geo,
        hl=hl,
        top_n=top_n,
    )


# -------------------------------------------------
# Metrics helpers
# -------------------------------------------------

PREFERRED_METRICS_ORDER = ["smape", "mape", "mase", "rmse", "mae"]


def _is_valid_metric_value(v) -> bool:
    import math
    if v is None:
        return False
    try:
        x = float(v)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(x) or math.isinf(x))


def _auto_choose_metric(sample_row: dict) -> str:
    for m in PREFERRED_METRICS_ORDER:
        if m in sample_row and _is_valid_metric_value(sample_row.get(m)):
            return m
    return "rmse"


# -------------------------------------------------
# Main AUTO mode
# -------------------------------------------------

def suggest_auto(
    base_keyword: str,
    top_n_related: int = 10,
    metric_name: str = "auto",
    outdir: str = "reports",
    collect_missing: bool = True,
    prediction_days: int = 30,
    seq_length: int = 12,
) -> List[Dict]:
    """
    AUTO-режим:
    1) related queries
    2) collect (якщо потрібно)
    3) report
    4) metrics
    5) ранжування
    """

    # lazy imports (не тягнемо TF при import модуля)
    from modules.core.fetch import run as run_collect
    from modules.report.pipeline import run as run_report

    ensure_dir(outdir)
    alt_dir = os.path.join(outdir, "_alt_eval")
    ensure_dir(alt_dir)

    related = get_related_keywords(base_keyword, top_n=top_n_related)
    if not related:
        print(f"[SUGGEST] Для '{base_keyword}' не знайдено related queries.")
        return []

    print(f"[SUGGEST] Related для '{base_keyword}':")
    for q in related:
        print(" -", q)

    results: List[Dict] = []

    for kw in related:
        print(f"\n[SUGGEST] Обробка '{kw}'")

        metrics_path = os.path.join(outdir, kw.replace(" ", "_") + "_metrics.json")

        # ---- collect ----
        if collect_missing and not os.path.exists(metrics_path):
            try:
                run_collect(db_path="db.sqlite3", keywords=[kw])
            except RequestException as e:
                print(f"[WARN] collect пропущено для '{kw}' (мережа/DNS): {e}")
                continue
            except Exception as e:
                print(f"[WARN] collect пропущено для '{kw}' (помилка): {e}")
                continue

        # ---- report ----
        code = run_report(
            kw,
            outdir=outdir,
            prediction_days=prediction_days,
            seq_length=seq_length,
        )
        if code != 0:
            print(f"[WARN] report для '{kw}' завершився з кодом {code}")
            continue

        if not os.path.exists(metrics_path):
            print(f"[WARN] Не знайдено {metrics_path}")
            continue

        with open(metrics_path, "r", encoding="utf-8") as f:
            met = json.load(f)

        met["_keyword"] = kw
        results.append(met)

    if not results:
        print("[SUGGEST] Немає жодного валідного результату.")
        return []

    # ---- metric auto select ----
    if metric_name == "auto":
        metric_name = _auto_choose_metric(results[0])
        print(
            f"[INFO] Обрана метрика: {metric_name} "
            f"(пріоритет: {', '.join(PREFERRED_METRICS_ORDER)})"
        )

    def sort_key(row):
        v = row.get(metric_name)
        if not _is_valid_metric_value(v):
            return float("inf")
        return float(v)

    results.sort(key=sort_key)

    # ---- table ----
    metrics_cols = [m for m in PREFERRED_METRICS_ORDER if any(m in r for r in results)]

    table = PrettyTable()
    table.field_names = ["keyword"] + metrics_cols

    for row in results:
        table.add_row([row["_keyword"]] + [row.get(m, "") for m in metrics_cols])

    print("\n=== Таблиця альтернатив для seed-слова:", base_keyword, "===")
    print(table)

    print(f"\n[BEST by {metric_name}] {results[0]['_keyword']}")

    # ---- save summary ----
    summary_path = os.path.join(
        alt_dir,
        f"{base_keyword.replace(' ', '_')}_summary_auto.json"
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metric_used": metric_name,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[SUGGEST] Зведення збережено: {summary_path}")
    return results
