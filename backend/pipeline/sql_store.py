import pandas as pd
import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PROC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "processed")
DB_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloudoptima.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")    # better concurrent read performance
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def build_resources_table(df: pd.DataFrame, conn: sqlite3.Connection):
    log.info("Building resources table ...")
    resources = (
        df.drop_duplicates("resource_id")
        [[
            "resource_id", "company", "account_id", "cloud_provider",
            "service", "instance_type", "region", "environment", "team",
            "is_reserved", "storage_gb", "waste_label"
        ]]
        .copy()
    )
    resources.to_sql("resources", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_res_company ON resources(company)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_res_waste   ON resources(waste_label)")
    log.info(f"  resources: {len(resources)} rows")


def build_daily_costs_table(df: pd.DataFrame, conn: sqlite3.Connection):
    log.info("Building daily_costs table ...")
    cols = [
        "date", "resource_id", "company", "cloud_provider", "service",
        "region", "environment", "team",
        "usage_hours", "cpu_utilisation", "memory_utilisation",
        "get_requests", "cost_usd", "waste_score", "waste_label",
        "cost_per_hour", "rolling_7d_cost", "rolling_30d_cost", "cost_vs_30d_avg",
        "idle_ratio", "cpu_7d_avg", "mem_7d_avg", "overprov_signal",
        "util_efficiency", "weekend_cost", "reservation_gap",
        "zombie_signal", "zero_req_7d_sum",
        "year", "month", "day_of_week", "is_weekend", "quarter"
    ]
    daily = df[cols].copy()
    daily.to_sql("daily_costs", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dc_date     ON daily_costs(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dc_resource ON daily_costs(resource_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dc_company  ON daily_costs(company)")
    log.info(f"  daily_costs: {len(daily):,} rows")


def build_monthly_summary(df: pd.DataFrame, conn: sqlite3.Connection):
    log.info("Building monthly_summary table ...")
    monthly = (
        df.groupby(["company", "cloud_provider", "service", "year", "month"])
        .agg(
            total_cost      = ("cost_usd",   "sum"),
            total_waste_cost= ("cost_usd",   lambda x: x[df.loc[x.index, "waste_label"] != "normal"].sum()),
            avg_cpu         = ("cpu_utilisation",    "mean"),
            avg_memory      = ("memory_utilisation", "mean"),
            active_resources= ("resource_id","nunique"),
        )
        .round(4)
        .reset_index()
    )
    monthly["waste_pct"] = (monthly["total_waste_cost"] / monthly["total_cost"].clip(lower=0.01) * 100).round(2)
    monthly.to_sql("monthly_summary", conn, if_exists="replace", index=False)
    log.info(f"  monthly_summary: {len(monthly)} rows")


def build_waste_summary(df: pd.DataFrame, conn: sqlite3.Connection):
    log.info("Building waste_summary table ...")
    waste = (
        df.groupby(["resource_id", "company", "cloud_provider", "service", "environment", "team", "waste_label"])
        .agg(
            total_cost        = ("cost_usd",           "sum"),
            avg_daily_cost    = ("cost_usd",           "mean"),
            avg_cpu           = ("cpu_utilisation",    "mean"),
            avg_memory        = ("memory_utilisation", "mean"),
            avg_waste_score   = ("waste_score",        "mean"),
            days_active       = ("date",               "count"),
            zombie_days       = ("zombie_signal",      "sum"),
            overprov_days     = ("overprov_signal",    "sum"),
            zero_req_days     = ("zero_request_flag",  "sum"),
            weekend_cost      = ("weekend_cost",       "sum"),
            last_seen         = ("date",               "max"),
        )
        .round(4)
        .reset_index()
    )
    # Potential monthly savings — heuristic by waste type
    waste["est_monthly_savings"] = waste.apply(_estimate_savings, axis=1).round(2)
    waste.to_sql("waste_summary", conn, if_exists="replace", index=False)
    log.info(f"  waste_summary: {len(waste)} rows")


def _estimate_savings(row) -> float:
    """
    Rule-based savings estimator — each waste type has a savings multiplier.
    This powers the Recommendations page in the dashboard.
    """
    label      = row["waste_label"]
    daily_cost = row["avg_daily_cost"]
    if label == "zombie_instance":
        return daily_cost * 30 * 0.95       # terminate → 95% savings
    elif label == "overprovisioned":
        return daily_cost * 30 * 0.40       # right-size → 40% savings
    elif label == "idle_storage":
        return daily_cost * 30 * 0.90       # delete/archive → 90% savings
    elif label == "reservation_waste":
        return daily_cost * 30 * 0.30       # resize reservation → 30%
    elif label == "dev_env_leak":
        return row["weekend_cost"] * (52/12) # stop weekends → full weekend cost savings
    return 0.0


def run():
    log.info(f"Loading billing_features.csv ...")
    df = pd.read_csv(os.path.join(PROC_DIR, "billing_features.csv"), parse_dates=["date"])
    # Add zero_request_flag if missing
    if "zero_request_flag" not in df.columns:
        df["zero_request_flag"] = (df["get_requests"] == 0).astype(int)

    conn = get_connection()
    build_resources_table(df, conn)
    build_daily_costs_table(df, conn)
    build_monthly_summary(df, conn)
    build_waste_summary(df, conn)
    conn.commit()
    conn.close()

    db_size = os.path.getsize(DB_PATH) / 1024 / 1024
    log.info(f"\nDatabase ready → {DB_PATH}  ({db_size:.1f} MB)")
    log.info("Tables: resources | daily_costs | monthly_summary | waste_summary")


if __name__ == "__main__":
    run()