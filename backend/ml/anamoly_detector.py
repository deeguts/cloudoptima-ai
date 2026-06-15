import pandas as pd
import numpy as np
import sqlite3
import os
import logging
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROC_DIR  = os.path.join(BASE_DIR, "..", "..", "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "..", "..", "data", "models")
DB_PATH   = os.path.join(BASE_DIR, "..", "database", "cloudoptima.db")
os.makedirs(MODEL_DIR, exist_ok=True)

# Features fed to Isolation Forest
IF_FEATURES = [
    "cpu_7d_avg",
    "mem_7d_avg",
    "cost_vs_30d_avg",
    "idle_ratio",
    "zombie_signal",
    "zero_req_7d_sum",
    "overprov_signal",
    "util_efficiency",
    "reservation_gap",
    "waste_score",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(PROC_DIR, "billing_features.csv"), parse_dates=["date"])
    log.info(f"Loaded {len(df):,} rows for anomaly detection")
    return df


def aggregate_to_resource_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Isolation Forest is run at resource level (one vector per resource),
    not at daily level. We aggregate the last 30 days of each resource
    to get a stable signal.
    """
    latest_date = df["date"].max()
    recent = df[df["date"] >= latest_date - pd.Timedelta(days=30)].copy()

    agg = recent.groupby("resource_id").agg(
        cpu_7d_avg        = ("cpu_utilisation",    "mean"),
        mem_7d_avg        = ("memory_utilisation", "mean"),
        cost_vs_30d_avg   = ("cost_vs_30d_avg",    "mean"),
        idle_ratio        = ("idle_ratio",          "mean"),
        zombie_signal     = ("zombie_signal",       "mean"),
        zero_req_7d_sum   = ("zero_req_7d_sum",     "mean"),
        overprov_signal   = ("overprov_signal",     "mean"),
        util_efficiency   = ("util_efficiency",     "mean"),
        reservation_gap   = ("reservation_gap",     "mean"),
        waste_score       = ("waste_score",         "mean"),
        avg_daily_cost    = ("cost_usd",            "mean"),
        total_30d_cost    = ("cost_usd",            "sum"),
        company           = ("company",             "first"),
        cloud_provider    = ("cloud_provider",      "first"),
        service           = ("service",             "first"),
        environment       = ("environment",         "first"),
        team              = ("team",                "first"),
        waste_label       = ("waste_label",         "first"),
    ).reset_index()

    log.info(f"Aggregated to {len(agg)} resource-level vectors")
    return agg


def train_and_score(agg: pd.DataFrame) -> pd.DataFrame:
    X = agg[IF_FEATURES].copy()

    # RobustScaler: handles outliers better than StandardScaler for billing data
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest — contamination reflects expected ~35% waste rate
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.35,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_scaled)

    # score_samples returns negative values: more negative = more anomalous
    raw_scores = iso.score_samples(X_scaled)

    # Normalise to [0, 1] where 1 = most anomalous
    min_s, max_s = raw_scores.min(), raw_scores.max()
    agg["anomaly_score"] = ((raw_scores - max_s) / (min_s - max_s + 1e-9)).round(4)

    # Binary flag at threshold 0.5
    agg["is_anomaly"] = (agg["anomaly_score"] >= 0.50).astype(int)

    # Save model + scaler
    joblib.dump(iso,    os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "if_scaler.pkl"))
    log.info(f"Model saved to {MODEL_DIR}")

    n_anomalies = agg["is_anomaly"].sum()
    log.info(f"Anomalies detected: {n_anomalies} / {len(agg)} resources ({n_anomalies/len(agg)*100:.1f}%)")
    log.info(f"Avg anomaly score by waste_label:")
    log.info(agg.groupby("waste_label")["anomaly_score"].mean().round(3).to_string())

    return agg


def save_results(agg: pd.DataFrame):
    out_csv = os.path.join(PROC_DIR, "anomaly_scores.csv")
    agg.to_csv(out_csv, index=False)
    log.info(f"Saved → {out_csv}")

    conn = sqlite3.connect(DB_PATH)
    agg.to_sql("anomaly_scores", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_anom_company ON anomaly_scores(company)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_anom_score   ON anomaly_scores(anomaly_score)")
    conn.commit()
    conn.close()
    log.info("Saved → anomaly_scores table in SQLite")


def run() -> pd.DataFrame:
    log.info("=" * 55)
    log.info("  CloudOptima AI — Anomaly Detector")
    log.info("=" * 55)
    df  = load_data()
    agg = aggregate_to_resource_level(df)
    agg = train_and_score(agg)
    save_results(agg)
    return agg


if __name__ == "__main__":
    run()