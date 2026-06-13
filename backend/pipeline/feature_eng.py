import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PROC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "processed")


def add_cost_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Adding cost features ...")

    df["cost_per_hour"] = np.where(
        df["usage_hours"] > 0,
        df["cost_usd"] / df["usage_hours"],
        0.0
    ).round(6)

    # Rolling means per resource 
    df = df.sort_values(["resource_id", "date"])
    df["rolling_7d_cost"]  = df.groupby("resource_id")["cost_usd"].transform(
        lambda x: x.rolling(7,  min_periods=1).mean()
    ).round(4)
    df["rolling_30d_cost"] = df.groupby("resource_id")["cost_usd"].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    ).round(4)

    df["cost_vs_30d_avg"] = np.where(
        df["rolling_30d_cost"] > 0,
        (df["cost_usd"] / df["rolling_30d_cost"]).round(4),
        1.0
    )

    return df


def add_utilisation_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Adding utilisation features ...")

    df["idle_ratio"] = np.where(
        df["usage_hours"] > 0,
        np.clip(1 - (df["cpu_utilisation"] / 100), 0, 1),
        0.0
    ).round(4)

    df["cpu_7d_avg"] = df.groupby("resource_id")["cpu_utilisation"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    ).round(2)
    df["mem_7d_avg"] = df.groupby("resource_id")["memory_utilisation"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    ).round(2)

    # Overprovisioning signal: cpu p95 < 30 AND mem p95 < 40 within rolling 30d window
    df["overprov_signal"] = (
        (df["cpu_7d_avg"]  < 30) &
        (df["mem_7d_avg"]  < 40) &
        (df["usage_hours"] > 0)
    ).astype(int)

    # Harmonic mean of utilisation (penalises imbalance)
    cpu_safe = df["cpu_utilisation"].clip(lower=0.1)
    mem_safe = df["memory_utilisation"].clip(lower=0.1)
    df["util_efficiency"] = (2 * cpu_safe * mem_safe / (cpu_safe + mem_safe)).round(2)

    return df

def add_waste_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Adding waste-specific features ...")

    # Weekend running cost
    df["weekend_cost"] = np.where(df["is_weekend"], df["cost_usd"], 0.0).round(4)

    # Reservation gap: reserved but usage is less than 10 hours/day
    df["reservation_gap"] = (
        df["is_reserved"] & (df["usage_hours"] < 10)
    ).astype(int)

    # Zombie signal: 24h usage + CPU < 5
    df["zombie_signal"] = (
        (df["usage_hours"] >= 22) &
        (df["cpu_utilisation"] < 5)
    ).astype(int)

    # Consecutive zero-request days per resource (rolling 7d)
    df["zero_request_flag"] = (df["get_requests"] == 0).astype(int)
    df["zero_req_7d_sum"]   = df.groupby("resource_id")["zero_request_flag"].transform(
        lambda x: x.rolling(7, min_periods=1).sum()
    ).astype(int)

    return df


def add_monthly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Adds per-resource monthly cost for trend analysis."""
    log.info("Adding monthly aggregates ...")
    monthly = (
        df.groupby(["resource_id", "year", "month"])["cost_usd"]
        .sum()
        .reset_index()
        .rename(columns={"cost_usd": "monthly_cost"})
    )
    df = df.merge(monthly, on=["resource_id", "year", "month"], how="left")
    return df


def run():
    log.info("Loading billing_clean.csv ...")
    df = pd.read_csv(os.path.join(PROC_DIR, "billing_clean.csv"), parse_dates=["date"])

    df = add_cost_features(df)
    df = add_utilisation_features(df)
    df = add_waste_features(df)
    df = add_monthly_aggregates(df)

    out_path = os.path.join(PROC_DIR, "billing_features.csv")
    df.to_csv(out_path, index=False)
    log.info(f"Feature engineering complete → {out_path}")
    log.info(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    run()