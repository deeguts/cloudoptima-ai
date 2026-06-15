import pandas as pd
import numpy as np
import sqlite3
import os
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

from prophet import Prophet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROC_DIR = os.path.join(BASE_DIR, "..", "..", "data", "processed")
DB_PATH  = os.path.join(BASE_DIR, "..", "database", "cloudoptima.db")

FORECAST_DAYS = 90

def load_daily_spend(level: str = "company") -> dict:
    """
    Returns dict of {key: pd.DataFrame} with columns [ds, y]
    level = 'company' → one series per company
    level = 'service' → one series per company+service
    """
    df = pd.read_csv(os.path.join(PROC_DIR, "billing_clean.csv"), parse_dates=["date"])

    series_dict = {}
    if level == "company":
        for company, grp in df.groupby("company"):
            daily = grp.groupby("date")["cost_usd"].sum().reset_index()
            daily.columns = ["ds", "y"]
            series_dict[company] = daily

    elif level == "service":
        for (company, service), grp in df.groupby(["company", "service"]):
            daily = grp.groupby("date")["cost_usd"].sum().reset_index()
            daily.columns = ["ds", "y"]
            if len(daily) >= 30:   # need at least 30 data points
                series_dict[f"{company}__{service}"] = daily

    log.info(f"Loaded {len(series_dict)} time series at '{level}' level")
    return series_dict


def build_prophet_model() -> Prophet:
    """
    Configured for Indian SaaS billing patterns:
      - Weekly seasonality: weekends lower spend
      - Monthly seasonality: month-end spikes
      - Yearly seasonality: Q1 budget flush
    """
    model = Prophet(
        changepoint_prior_scale  = 0.15,    
        seasonality_prior_scale  = 10.0,
        holidays_prior_scale     = 10.0,
        daily_seasonality        = False,
        weekly_seasonality       = True,
        yearly_seasonality       = True,
        interval_width           = 0.90,    # 90% confidence interval for budget alerts
    )
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    return model


def forecast_series(key: str, series: pd.DataFrame) -> pd.DataFrame:
    """Fits Prophet on one time series, returns forecast DataFrame."""
    model = build_prophet_model()
    model.fit(series)

    future   = model.make_future_dataframe(periods=FORECAST_DAYS)
    forecast = model.predict(future)

    # Keep relevant columns only
    out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend",
                    "weekly", "yearly"]].copy()
    out.columns = ["date", "forecast_cost", "forecast_lower", "forecast_upper",
                   "trend", "weekly_seasonality", "yearly_seasonality"]
    out["forecast_cost"]  = out["forecast_cost"].clip(lower=0).round(4)
    out["forecast_lower"] = out["forecast_lower"].clip(lower=0).round(4)
    out["forecast_upper"] = out["forecast_upper"].clip(lower=0).round(4)
    out["key"]            = key
    out["is_forecast"]    = out["date"] > series["ds"].max()
    return out


def run_all_forecasts(series_dict: dict, level: str) -> pd.DataFrame:
    all_forecasts = []
    for i, (key, series) in enumerate(series_dict.items()):
        log.info(f"  [{i+1}/{len(series_dict)}] Forecasting: {key}")
        try:
            fc = forecast_series(key, series)
            fc["level"] = level
            if level == "company":
                fc["company"] = key
                fc["service"] = "ALL"
            else:
                parts = key.split("__")
                fc["company"] = parts[0]
                fc["service"] = parts[1] if len(parts) > 1 else "unknown"
            all_forecasts.append(fc)
        except Exception as e:
            log.warning(f"  Failed {key}: {e}")

    return pd.concat(all_forecasts, ignore_index=True)


def compute_budget_alerts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    For each company, compute the month where predicted spend
    first exceeds the trailing 3-month average by >20%.
    This powers the 'Budget Alert' KPI card on the dashboard.
    """
    alerts = []
    for company, grp in forecasts[forecasts["level"] == "company"].groupby("company"):
        historical = grp[~grp["is_forecast"]].tail(90)
        avg_monthly = historical["forecast_cost"].sum() / 3

        future = grp[grp["is_forecast"]].copy()
        future["month"] = future["date"].dt.to_period("M")
        monthly_fc = future.groupby("month")["forecast_cost"].sum()

        for month, cost in monthly_fc.items():
            if cost > avg_monthly * 1.20:
                alerts.append({
                    "company":          company,
                    "alert_month":      str(month),
                    "forecast_cost":    round(cost, 2),
                    "baseline_monthly": round(avg_monthly, 2),
                    "overage_pct":      round((cost / avg_monthly - 1) * 100, 1),
                })
                break   # first breach only
    cols = ["company","alert_month","forecast_cost","baseline_monthly","overage_pct"]
    return pd.DataFrame(alerts, columns=cols) if alerts else pd.DataFrame(columns=cols)


def save_results(forecasts: pd.DataFrame, alerts: pd.DataFrame):
    fc_path = os.path.join(PROC_DIR, "forecasts.csv")
    al_path = os.path.join(PROC_DIR, "budget_alerts.csv")
    forecasts.to_csv(fc_path, index=False)
    alerts.to_csv(al_path,    index=False)
    log.info(f"Saved → {fc_path}")
    log.info(f"Saved → {al_path}")

    conn = sqlite3.connect(DB_PATH)
    forecasts.to_sql("forecasts",     conn, if_exists="replace", index=False)
    alerts.to_sql("budget_alerts",    conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fc_company ON forecasts(company)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fc_date    ON forecasts(date)")
    conn.commit()
    conn.close()
    log.info("Saved → forecasts + budget_alerts tables in SQLite")


def run():
    log.info("=" * 55)
    log.info("  CloudOptima AI — Prophet Forecaster")
    log.info("=" * 55)

    log.info("\nPhase A: Company-level forecasts")
    company_series   = load_daily_spend("company")
    company_fc       = run_all_forecasts(company_series, "company")

    log.info("\nPhase B: Service-level forecasts")
    service_series   = load_daily_spend("service")
    service_fc       = run_all_forecasts(service_series, "service")

    all_forecasts    = pd.concat([company_fc, service_fc], ignore_index=True)

    log.info("\nComputing budget alerts ...")
    alerts           = compute_budget_alerts(all_forecasts)

    save_results(all_forecasts, alerts)

    # Summary
    fc_only = all_forecasts[all_forecasts["is_forecast"] & (all_forecasts["level"] == "company")]
    log.info("\n90-day forecast summary (company level):")
    for company, grp in fc_only.groupby("company"):
        total = grp["forecast_cost"].sum()
        log.info(f"  {company:<20} ${total:>10,.2f}")
    if len(alerts):
        log.info(f"\nBudget alerts triggered: {len(alerts)}")
        log.info(alerts[["company","alert_month","overage_pct"]].to_string(index=False))

    return all_forecasts, alerts


if __name__ == "__main__":
    run()