"""
Rules applied:
  1. Parse and validate dates
  2. Clip utilisation to [0, 100]
  3. Clip cost to [0, ∞)
  4. Impute storage_gb nulls (non-storage resources → 0)
  5. Normalise string columns (lowercase, strip)
  6. Add month, year, day_of_week, is_weekend columns
  7. Flag and log any rows dropped
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

RAW_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

STORAGE_SERVICES = {"S3", "BlobStorage", "EBS", "ManagedDisks"}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    original_len = len(df)
    log.info(f"Starting cleaning — {original_len:,} rows")

    # types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates  = df["date"].isna().sum()
    if bad_dates:
        log.warning(f"Dropping {bad_dates} rows with unparseable dates")
    df = df.dropna(subset=["date"])

    #Clip utilisation 
    df["cpu_utilisation"]    = df["cpu_utilisation"].clip(0, 100)
    df["memory_utilisation"] = df["memory_utilisation"].clip(0, 100)
    df["usage_hours"]        = df["usage_hours"].clip(0, 24)

    #Clip cost 
    neg_cost = (df["cost_usd"] < 0).sum()
    if neg_cost:
        log.warning(f"Clipping {neg_cost} negative cost rows to 0")
    df["cost_usd"] = df["cost_usd"].clip(lower=0)

    # ── 4. Storage GB imputation 
    df["storage_gb"] = df.apply(
        lambda r: r["storage_gb"] if r["service"] in STORAGE_SERVICES else 0.0,
        axis=1
    )
    df["storage_gb"] = df["storage_gb"].fillna(0.0)

    #Normalise strings 
    str_cols = ["company", "cloud_provider", "service", "region", "environment", "team", "waste_label"]
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    #Temporal features 
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek   # 0=Mon, 6=Sun
    df["is_weekend"]  = df["day_of_week"] >= 5
    df["week_num"]    = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"]     = df["date"].dt.quarter

    # Sort
    df = df.sort_values(["company", "resource_id", "date"]).reset_index(drop=True)

    dropped = original_len - len(df)
    log.info(f"Cleaning complete — {len(df):,} rows kept, {dropped} dropped")
    return df


def run():
    log.info("Loading billing_combined.csv ...")
    df_raw = pd.read_csv(os.path.join(RAW_DIR, "billing_combined.csv"))
    df     = clean(df_raw)

    out_path = os.path.join(PROC_DIR, "billing_clean.csv")
    df.to_csv(out_path, index=False)
    log.info(f"Saved → {out_path}")
    return df


if __name__ == "__main__":
    run()