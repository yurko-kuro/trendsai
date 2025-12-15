# trendsai/cli.py

import argparse
from pathlib import Path

from modules.core.fetch import GoogleTrendsToSQLite
from modules.report.pipeline import run as save_report
from trendsai.semantic import semantic_rank
from trendsai.cluster_modes import load_cluster_keywords
from trendsai.compare import compare_cluster
from modules.alt.auto_suggest import suggest_auto


def build_cli():
    ap = argparse.ArgumentParser(
        prog="trendsai",
        description="TrendsAI — дипломний інструмент прогнозування",
    )

    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- collect ---
    p = sub.add_parser("collect", help="Завантажити дані Google Trends")
    p.add_argument("keywords", nargs="+")
    p.add_argument("--db", default="db.sqlite3")
    p.add_argument("--days", type=int, default=1800)

    # --- semantic ---
    p = sub.add_parser("semantic", help="Семантичний підбір альтернатив")
    p.add_argument("keyword")
    p.add_argument("--db", default="db.sqlite3")
    p.add_argument("--top", type=int, default=5)

    # --- report ---
    p = sub.add_parser("report", help="Створити PNG/CSV/JSON для ключового слова")
    p.add_argument("keyword")
    p.add_argument("--db", default="db.sqlite3")
    p.add_argument("--out", default="reports")

    # --- compare ---
    p = sub.add_parser("compare", help="Порівняти всі слова у кластері seed-слова")
    p.add_argument("keyword")
    p.add_argument(
        "--metric",
        default="auto",
        help="Метрика для порівняння (rmse, mae, mape, smape, mase або 'auto')",
    )
    p.add_argument("--reports", default="reports")
    p.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Додаткові ключові слова, які треба догрузити і включити в порівняння",
    )

    # --- auto ---
    p = sub.add_parser(
        "auto",
        help="Авто-підбір альтернатив через Google Trends і ранжування моделей",
    )
    p.add_argument("keyword")
    p.add_argument(
        "--top",
        type=int,
        default=10,
        help="Скільки related-слiв брати з Google Trends",
    )
    p.add_argument(
        "--metric",
        default="auto",
        help="Метрика для ранжування (rmse, mae, mape, smape, mase або 'auto')",
    )
    p.add_argument("--reports", default="reports")

    return ap


def run(argv=None):
    parser = build_cli()
    args = parser.parse_args(argv)

    # === ROUTING ===

    if args.cmd == "collect":
        gt = GoogleTrendsToSQLite(args.db, days=args.days)
        for kw in args.keywords:
            gt.process_keyword(kw)
        print("[OK] Збір завершено.")
        return 0

    if args.cmd == "semantic":
        cluster = load_cluster_keywords(args.db, args.keyword)
        ranked = semantic_rank(args.keyword, cluster)
        for kw, sc in ranked:
            print(f"{kw:25s} cos={sc:.3f}")
        return 0

    if args.cmd == "report":
        Path(args.out).mkdir(exist_ok=True)
        save_report(args.keyword, args.out, 30, 12)
        print("[OK] Артефакти збережено.")
        return 0

    if args.cmd == "compare":
        res = compare_cluster(
            seed=args.keyword,
            metric=args.metric,
            reports_dir=args.reports,
            extra_keywords=args.extra,
        )

        if res:
            print(f"\n[BEST] {res['best']}")
            print(f"[SAVED] {res['csv']}")
            print(f"[SAVED] {res['json']}")
        return 0

    if args.cmd == "auto":
        suggest_auto(
            base_keyword=args.keyword,
            top_n_related=args.top,
            metric_name=args.metric,
            outdir=args.reports,
            collect_missing=True,
            prediction_days=30,
            seq_length=12,
        )
        return 0

    return 0
