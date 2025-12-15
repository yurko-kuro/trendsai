# modules/alt/compare.py

import os
import json
from typing import Dict, List


def load_metrics_from_reports(keyword: str, reports_dir: str) -> Dict:
    """
    Завантажує метрики з *_metrics.json для одного слова.
    """
    base = keyword.replace(" ", "_")
    fpath = os.path.join(reports_dir, f"{base}_metrics.json")

    if not os.path.exists(fpath):
        return {}

    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_cluster_metrics(keywords: List[str], metric: str, reports_dir: str) -> List[Dict]:
    """
    Завантажує *_metrics.json для всіх слів кластера
    та повертає список словників, відсортований за метрикою.
    """
    result: List[Dict] = []

    for kw in keywords:
        met = load_metrics_from_reports(kw, reports_dir)
        if not met:
            continue
        met["_keyword"] = kw
        result.append(met)

    # менше значення метрики — краще
    result.sort(key=lambda x: x.get(metric, float("inf")))
    return result


def save_comparison_results(results: List[Dict], reports_dir: str, seed: str, metric: str) -> Dict:
    """
    Зберігає підсумкові CSV + JSON для порівняння кластера.
    """
    if not results:
        return {}

    base = seed.replace(" ", "_")
    csv_path = os.path.join(reports_dir, f"{base}_cluster_{metric}.csv")
    json_path = os.path.join(reports_dir, f"{base}_cluster_{metric}.json")

    # CSV
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("keyword," + metric + "\n")
        for row in results:
            f.write(f"{row['_keyword']},{row.get(metric, '')}\n")

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return {
        "csv": csv_path,
        "json": json_path,
        "best": results[0]["_keyword"],
    }
