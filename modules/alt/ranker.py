# modules/alt/ranker.py
from typing import List, Optional

import pandas as pd

from modules.alt.selector import TransformerKeywordSelector


def rank_alternatives(
    base_keyword: str,
    top_k: int = 10,
    forecast_days: int = 30,  # залишаємо для сумісності сигнатури
    db_path: str = "db.sqlite3",
    seq: int = 12,
    candidates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Повертає DataFrame з колонками:
      - keyword
      - similarity

    Якщо candidates передано — ранжуємо лише їх.
    Якщо ні — авто-пошук по всіх словах з БД (trends_keyword).
    """
    sel = TransformerKeywordSelector(db_path=db_path)
    df = sel.candidates(
        base_keyword=base_keyword,
        top_k=top_k,
        manual_candidates=candidates,
    )

    if df.empty:
        return pd.DataFrame(columns=["keyword", "similarity"])

    df = df.rename(columns={"name": "keyword"})
    return df[["keyword", "similarity"]]
