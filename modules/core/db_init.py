import sqlite3
import os


def init_db(db_path: str = "db.sqlite3"):
    """Створює файл БД (якщо немає) і піднімає всю схему."""
    # створити файл, якщо його ще нема
    if not os.path.exists(db_path):
        open(db_path, "w").close()

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        # ==============================
        #  trends_keyword
        # ==============================
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trends_keyword (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT UNIQUE NOT NULL,
                cluster_id INTEGER
            )
        """)

        # на випадок старої БД без cluster_id
        cur.execute("PRAGMA table_info(trends_keyword)")
        cols = [row[1] for row in cur.fetchall()]
        if "cluster_id" not in cols:
            cur.execute("ALTER TABLE trends_keyword ADD COLUMN cluster_id INTEGER")

        # ==============================
        #  trends_interest (часовий ряд)
        # ==============================
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trends_interest (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                date       TEXT NOT NULL,
                value      INTEGER,
                keyword_id INTEGER NOT NULL,
                is_partial INTEGER NOT NULL DEFAULT 0,
                UNIQUE(keyword_id, date),
                FOREIGN KEY (keyword_id) REFERENCES trends_keyword(id)
            )
        """)

        # ==============================
        #  metrics (зберігаємо оцінки)
        # ==============================
        cur.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword_id   INTEGER NOT NULL,
                metric_name  TEXT NOT NULL,
                metric_value REAL NOT NULL,
                created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (keyword_id) REFERENCES trends_keyword(id)
            )
        """)

        # додаємо cluster_id якщо його ще немає
        cur.execute("PRAGMA table_info(metrics)")
        mcols = [row[1] for row in cur.fetchall()]
        if "cluster_id" not in mcols:
            cur.execute("ALTER TABLE metrics ADD COLUMN cluster_id INTEGER")

        # ==============================
        # keyword_cluster (кластер)
        # ==============================
        cur.execute("""
            CREATE TABLE IF NOT EXISTS keyword_cluster (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                base_keyword TEXT NOT NULL,
                created_at   TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ==============================
        # keyword_semantic (зв’язки)
        # ==============================
        cur.execute("""
            CREATE TABLE IF NOT EXISTS keyword_semantic (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                base_keyword_id      INTEGER NOT NULL,
                candidate_keyword_id INTEGER NOT NULL,
                sim_score            REAL NOT NULL,
                created_at           TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (base_keyword_id, candidate_keyword_id),
                FOREIGN KEY (base_keyword_id)
                    REFERENCES trends_keyword(id) ON DELETE CASCADE,
                FOREIGN KEY (candidate_keyword_id)
                    REFERENCES trends_keyword(id) ON DELETE CASCADE
            )
        """)

        # ==============================
        # Індекси
        # ==============================
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_trends_interest_kw
            ON trends_interest(keyword_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_trends_keyword_name
            ON trends_keyword(name)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_trends_keyword_cluster
            ON trends_keyword(cluster_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_semantic_base
            ON keyword_semantic(base_keyword_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_semantic_candidate
            ON keyword_semantic(candidate_keyword_id)
        """)

        conn.commit()
