# trendsai/cluster_modes.py

import sqlite3
from modules.core.fetch import GoogleTrendsToSQLite
from trendsai.semantic import semantic_rank
from modules.alt.auto_suggest import discover_candidates_via_trends


def load_cluster_keywords(db_path: str, seed: str) -> list[str]:
    """
    Повертає всі слова з БД, які належать до того ж кластера, що й seed.
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        cur.execute("SELECT cluster_id FROM trends_keyword WHERE name = ?", (seed,))
        row = cur.fetchone()
        if not row or row[0] is None:
            return []

        cluster_id = row[0]

        cur.execute(
            "SELECT name FROM trends_keyword WHERE cluster_id = ? ORDER BY name",
            (cluster_id,)
        )
        return [r[0] for r in cur.fetchall()]
