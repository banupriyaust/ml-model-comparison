"""
Data loader for anomaly detection project.
Loads a stratified random sample from the claims SQLite database
and caches it as a parquet file for subsequent runs.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from anomaly_detection.config import DB_PATH, RESULTS_DIR, SAMPLE_SIZE, RANDOM_STATE


CACHE_PATH = RESULTS_DIR / "claims_sample.parquet"


def load_claims_sample(sample_size: int = SAMPLE_SIZE, use_cache: bool = True) -> pd.DataFrame:
    """
    Load a random sample of claims from the SQLite database.
    Caches result as parquet for fast reloads.
    """
    if use_cache and CACHE_PATH.exists():
        print(f"Loading cached sample from {CACHE_PATH}...")
        return pd.read_parquet(CACHE_PATH)

    print(f"Sampling {sample_size:,} rows from {DB_PATH}...")
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA cache_size=-200000")  # 200MB cache

    query = f"""
        SELECT *
        FROM claims
        ORDER BY RANDOM()
        LIMIT {sample_size}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")

    # Cache for fast reloads
    df.to_parquet(CACHE_PATH, index=False)
    print(f"Cached sample to {CACHE_PATH}")

    return df
