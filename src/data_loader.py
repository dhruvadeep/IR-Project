from itertools import islice
from pathlib import Path

import duckdb
from datasets import load_dataset

DB_PATH = Path("data/ccnews.db")
DB_PATH.parent.mkdir(exist_ok=True)


def load_and_clean(limit=10000):
    """Load dataset, clean, and store in DuckDB"""
    con = duckdb.connect(str(DB_PATH))

    # 1. Create RAW table (No IDs yet, just dump data)
    con.execute("""
    CREATE TABLE IF NOT EXISTS news (
        title TEXT,
        body TEXT,
        date TEXT,
        site TEXT,
        language TEXT
    );
    """)

    # 2. Load dataset
    print("Loading dataset...")
    ds = load_dataset(
        "stanford-oval/ccnews", name="2023", split="train", streaming=True
    )

    # Filter for English only
    ds = (x for x in ds if x["language"] == "en")
    sample = list(islice(ds, limit))

    # 3. Insert into raw table
    print(f"Inserting {len(sample)} articles...")
    batch = [
        (
            x["title"],
            x["plain_text"],
            x["published_date"],
            x["sitename"],
            x["language"],
        )
        for x in sample
    ]
    con.executemany(
        "INSERT INTO news (title, body, date, site, language) VALUES (?, ?, ?, ?, ?)",
        batch,
    )

    # 4. Clean & Generate IDs
    # We dedup on content FIRST, then generate IDs using row_number()
    print("Cleaning and generating IDs...")
    con.execute("""
    CREATE OR REPLACE TABLE clean_news AS
    SELECT 
        row_number() OVER () as doc_id,
        title, 
        body, 
        date, 
        site, 
        language
    FROM (
        SELECT DISTINCT title, body, date, site, language
        FROM news
        WHERE title IS NOT NULL AND body IS NOT NULL
    );
    """)

    count = con.execute("SELECT COUNT(*) FROM clean_news").fetchone()[0]
    print(f"✓ Cleaned dataset: {count} unique articles")

    con.close()


def get_docs():
    """Fetch all clean docs from DB"""
    con = duckdb.connect(str(DB_PATH))
    docs = con.execute(
        "SELECT doc_id, title, body, site, date FROM clean_news ORDER BY doc_id"
    ).fetchall()
    con.close()
    return docs
