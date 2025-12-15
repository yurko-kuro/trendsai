import sqlite3
from modules.core.db_init import init_db
import argparse
import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
import logging
import time
import warnings
from datetime import datetime, timedelta
from tqdm import tqdm
from requests.exceptions import ReadTimeout

# Очищення файлу логів перед записом
open("trends.log", "w").close()

warnings.simplefilter(action="ignore", category=FutureWarning)

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    filename="trends.log",
    filemode="a",
    encoding="utf-8",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class GoogleTrendsToSQLite:
    def __init__(
        self,
        db_path: str = "db.sqlite3",
        timeout: int = 60,
        attempts: int = 5,
        days: int = 30,
    ):
        """
        timeout  – таймаут HTTP-запиту до Google (секунди)
        attempts – кількість спроб при помилках (rate limit / timeout)
        days     – за скільки днів тягнути історію
        """
        self.db_path = db_path
        self.timeout = timeout
        self.attempts = attempts
        self.days = days

        # TrendReq з таймаутом
        self.pytrends = TrendReq(timeout=self.timeout)

        # Створити файл БД + підняти схему
        init_db(self.db_path)

    # ===================== робота з БД =====================

    def save_keyword(self, keyword: str) -> int:
        """Зберегти ключове слово в таблиці trends_keyword і повернути його id."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO trends_keyword (name) VALUES (?)",
                (keyword,),
            )
            conn.commit()
            cursor.execute(
                "SELECT id FROM trends_keyword WHERE name = ?",
                (keyword,),
            )
            row = cursor.fetchone()
        return row[0]

    def save_interest_data(self, keyword_id: int, data: pd.DataFrame) -> None:
        """Зберегти / оновити дані про інтерес до ключового слова."""
        if data is None or data.empty:
            logging.info(f"[DB] Порожні дані для keyword_id={keyword_id}, нічого зберігати.")
            return

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for _, row in data.iterrows():
                cursor.execute(
                    """
                    INSERT INTO trends_interest (keyword_id, date, value, is_partial)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(keyword_id, date) DO UPDATE SET
                        value = excluded.value,
                        is_partial = excluded.is_partial
                    """,
                    (
                        keyword_id,
                        row["date"].strftime("%Y-%m-%d"),
                        int(row["value"]),
                        int(row["isPartial"]),
                    ),
                )
            conn.commit()

    # ===================== робота з Google Trends =====================

    def fetch_interest_over_time(self, keyword: str, timeframe: str) -> pd.DataFrame:
        """
        Отримання даних для keyword у вказаному timeframe.
        Детально логірує всі спроби, помилки та очікування.
        """
        attempt = 0
        last_error = None

        while attempt < self.attempts:
            attempt += 1
            logging.info(
                f"[FETCH] attempt={attempt}/{self.attempts} "
                f"keyword='{keyword}' timeframe='{timeframe}'"
            )

            try:
                start_ts = time.time()
                self.pytrends.build_payload(
                    [keyword],
                    cat=0,
                    timeframe=timeframe,
                    geo="UA",
                    gprop="",
                )
                data = self.pytrends.interest_over_time()
                elapsed = time.time() - start_ts

                if data is None or data.empty:
                    logging.warning(
                        f"[FETCH] empty dataframe "
                        f"keyword='{keyword}' timeframe='{timeframe}' "
                        f"elapsed={elapsed:.2f}s"
                    )
                    # порожня відповідь – немає сенсу ретраїти
                    return pd.DataFrame()

                data = data.reset_index().rename(
                    columns={keyword: "value", "isPartial": "isPartial"}
                )
                data = data[["date", "value", "isPartial"]]

                logging.info(
                    f"[FETCH] success keyword='{keyword}' timeframe='{timeframe}' "
                    f"rows={len(data)} elapsed={elapsed:.2f}s"
                )
                return data

            except TooManyRequestsError as e:
                last_error = e
                wait_time = (2 ** (attempt - 1)) * 60
                logging.warning(
                    f"[RATE_LIMIT] TooManyRequestsError for keyword='{keyword}' "
                    f"timeframe='{timeframe}', attempt={attempt}/{self.attempts}, "
                    f"sleep={wait_time}s, error={e}"
                )
                time.sleep(wait_time)

            except ReadTimeout as e:
                last_error = e
                wait_time = (2 ** (attempt - 1)) * 60
                logging.warning(
                    f"[TIMEOUT] ReadTimeout for keyword='{keyword}' "
                    f"timeframe='{timeframe}', attempt={attempt}/{self.attempts}, "
                    f"sleep={wait_time}s, error={e}"
                )
                time.sleep(wait_time)

            except Exception as e:
                last_error = e
                logging.exception(
                    f"[FETCH] unexpected error keyword='{keyword}' "
                    f"timeframe='{timeframe}', attempt={attempt}/{self.attempts}: {e}"
                )
                # при несподіваній помилці немає сенсу продовжувати
                break

        logging.error(
            f"[FETCH] FAILED after {self.attempts} attempts "
            f"keyword='{keyword}' timeframe='{timeframe}', last_error={last_error}"
        )
        return pd.DataFrame()

    # ===================== допоміжні методи для поблочного збору =====================

    def _fetch_range(
        self,
        keyword: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        """
        Обгортає fetch_interest_over_time для інтервалу [start_dt; end_dt].
        """
        timeframe = f"{start_dt.strftime('%Y-%m-%d')} {end_dt.strftime('%Y-%m-%d')}"
        logging.info(
            f"[RANGE] keyword='{keyword}' {start_dt.date()}–{end_dt.date()} "
            f"timeframe='{timeframe}'"
        )
        return self.fetch_interest_over_time(keyword, timeframe)

    def _save_data(self, keyword_id: int, df: pd.DataFrame) -> None:
        """
        Перевіряє dataframe і зберігає його через save_interest_data.
        """
        if df is None or df.empty:
            logging.info(
                f"[RANGE] Порожні дані для keyword_id={keyword_id}, блок пропущено."
            )
            return
        self.save_interest_data(keyword_id, df)

    # ===================== оновлення partial-значень =====================

    def update_partial_data(self, keyword_id: int, keyword: str) -> None:
        """
        Оновлення записів, де is_partial = 1.
        Для кожної дати робимо окремий короткий запит і намагаємось замінити
        часткове значення на фінальне.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT date FROM trends_interest
                WHERE keyword_id = ? AND is_partial = 1
                """,
                (keyword_id,),
            )
            partial_dates = cursor.fetchall()

        if not partial_dates:
            logging.info(f"[PARTIAL] Не знайдено часткових даних для '{keyword}'.")
            return

        logging.info(
            f"[PARTIAL] Found {len(partial_dates)} partial records for '{keyword}'. "
            f"Перевіряємо оновлені значення..."
        )

        for (date_str,) in partial_dates:
            timeframe = f"{date_str} {date_str}"
            logging.info(
                f"[PARTIAL] Refresh keyword='{keyword}' date={date_str} timeframe='{timeframe}'"
            )

            new_data = self.fetch_interest_over_time(keyword, timeframe)
            if new_data.empty:
                logging.info(
                    f"[PARTIAL] Still empty/partial for '{keyword}' date={date_str}"
                )
                continue

            row0 = new_data.iloc[0]
            if row0["isPartial"] == 0:
                new_value = int(row0["value"])
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        UPDATE trends_interest
                        SET value = ?, is_partial = 0
                        WHERE keyword_id = ? AND date = ?
                        """,
                        (new_value, keyword_id, date_str),
                    )
                    conn.commit()
                logging.info(
                    f"[PARTIAL] Updated '{keyword}' date={date_str} value={new_value}"
                )
            else:
                logging.info(
                    f"[PARTIAL] Value for '{keyword}' date={date_str} "
                    f"залишається partial (is_partial=1)"
                )

    # ===================== основний цикл для одного слова =====================

    def process_keyword(self, keyword: str) -> None:
        """
        Повний цикл обробки одного ключового слова:
          - створення/отримання keyword_id
          - поблочне завантаження історії (по ~30 днів)
          - оновлення partial-значень
        """
        logging.info(f"[PROCESS] start keyword='{keyword}', days={self.days}")
        keyword_id = self.save_keyword(keyword)

        end_date = datetime.today()
        start_date = end_date - timedelta(days=self.days)
        total_days = (end_date - start_date).days
        total_periods = max(total_days // 30, 1)

        with tqdm(total=total_periods, desc=f"Ключове слово: {keyword}") as pbar:
            cur_start = start_date
            while cur_start < end_date:
                cur_end = cur_start + timedelta(days=30)
                if cur_end > end_date:
                    cur_end = end_date

                try:
                    df = self._fetch_range(keyword, cur_start, cur_end)
                    self._save_data(keyword_id, df)
                except Exception as e:
                    logging.warning(
                        f"Помилка при завантаженні {cur_start} – {cur_end}: {e}"
                    )

                cur_start = cur_end
                pbar.update(1)

        # після поблочного завантаження — спробувати оновити partial-рядки
        self.update_partial_data(keyword_id, keyword)
        logging.info(f"[PROCESS] finished keyword='{keyword}'")


# ===================== простий CLI для окремого запуску =====================

def main():
    parser = argparse.ArgumentParser(
        description="Збір даних Google Trends у SQLite"
    )
    parser.add_argument(
        "keywords",
        nargs="+",
        help="Ключові слова (одне або декілька)",
    )
    parser.add_argument(
        "--db",
        default="db.sqlite3",
        help="Шлях до файлу бази даних",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1800,
        help="За скільки днів тягнути історію (за замовчуванням 1800)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Таймаут HTTP-запиту до Google, с",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=5,
        help="Кількість спроб при помилках/таймаутах",
    )

    args = parser.parse_args()

    gt = GoogleTrendsToSQLite(
        db_path=args.db,
        timeout=args.timeout,
        attempts=args.attempts,
        days=args.days,
    )

    for kw in args.keywords:
        gt.process_keyword(kw)

    print("[OK] Збір завершено.")

# ============================================================
# Фасад для сумісності з auto_suggest та іншими модулями
# ============================================================

def run(db_path="db.sqlite3", keywords=None, days=1800, timeout=60, attempts=5):
    """
    Простий універсальний запуск збирання даних.
    Використовується у modules.alt.auto_suggest.
    """
    if not keywords:
        return

    gt = GoogleTrendsToSQLite(
        db_path=db_path,
        timeout=timeout,
        attempts=attempts,
        days=days,
    )

    for kw in keywords:
        gt.process_keyword(kw)

if __name__ == "__main__":
    main()
