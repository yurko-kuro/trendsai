# modules/alt/selector.py
import sqlite3
from typing import List, Optional

import pandas as pd
from sentence_transformers import SentenceTransformer, util


class TransformerKeywordSelector:
    """
    Семантичний відбір кандидатів за допомогою Sentence-BERT.
    Джерело кандидатів:
      - усі ключові слова з таблиці trends_keyword (авто-режим),
      - або явно переданий список (ручний режим).
    """

    def __init__(
        self,
        db_path: str = "db.sqlite3",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        self.db_path = db_path
        # модель вантажиться один раз
        self.model = SentenceTransformer(model_name)

    def _load_all_keywords(self) -> List[str]:
        """Зчитати всі ключові слова з БД."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(
                "SELECT DISTINCT name FROM trends_keyword ORDER BY name",
                conn
            )
        return df["name"].tolist()

    def candidates(
        self,
        base_keyword: str,
        top_k: int = 10,
        min_similarity: float = 0.5,
        manual_candidates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Повертає DataFrame з колонками:
          - name        (альтернативне ключове слово)
          - similarity  (косинусна подібність до base_keyword)

        Якщо manual_candidates не None — працюємо тільки з ними.
        Якщо None — беремо всі name з trends_keyword.
        """
        base_keyword = base_keyword.strip()
        if not base_keyword:
            return pd.DataFrame(columns=["name", "similarity"])

        if manual_candidates is not None:
            pool = [w.strip() for w in manual_candidates if w.strip()]
        else:
            pool = self._load_all_keywords()

        # прибрати базове слово і дублікати
        pool = sorted({w for w in pool if w and w != base_keyword})
        if not pool:
            return pd.DataFrame(columns=["name", "similarity"])

        # ембеддинги
        emb_base = self.model.encode(base_keyword, convert_to_tensor=True)
        emb_pool = self.model.encode(pool, convert_to_tensor=True)

        cos_scores = util.cos_sim(emb_base, emb_pool)[0].cpu().tolist()
        df = pd.DataFrame({
            "name": pool,
            "similarity": cos_scores
        })

        # фільтр за порогом, сортування, top_k
        df = df[df["similarity"] >= min_similarity].sort_values(
            "similarity", ascending=False
        )
        if top_k is not None and top_k > 0:
            df = df.head(top_k)

        df.reset_index(drop=True, inplace=True)
        return df
