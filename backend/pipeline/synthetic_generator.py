import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import hashlib

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

END_DATE   = datetime(2025, 6, 30)
START_DATE = END_DATE - timedelta(days=548)
DATE_RANGE = pd.date_range(START_DATE, END_DATE, freq="D")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
os.makedirs(OUT_DIR, exist_ok=True)

COMPANIES = {
    "zylara": {
        "account_id":    "aws-zylara-981234",
        "cloud":         "AWS",
        "city":          "Mumbai",
        "size":          "mid",
        "growth_rate":   0.008,
        "waste_profile": {
            "zombie_pct":        0.12,
            "overprov_pct":      0.25,
            "idle_storage_pct":  0.18,
            "reservation_waste": 0.10,
            "dev_leak_pct":      0.20,
        },
    },
    "kratos_cloud": {
        "account_id":    "aws-kratos-445521",
        "cloud":         "AWS",
        "city":          "Pune",
        "size":          "large",
        "growth_rate":   0.012,
        "waste_profile": {
            "zombie_pct":        0.08,
            "overprov_pct":      0.30,
            "idle_storage_pct":  0.25,
            "reservation_waste": 0.20,
            "dev_leak_pct":      0.15,
        },
    },
    "novapay": {
        "account_id":    "azure-novapay-772891",
        "cloud":         "Azure",
        "city":          "Bengaluru",
        "size":          "mid",
        "growth_rate":   0.015,
        "waste_profile": {
            "zombie_pct":        0.15,
            "overprov_pct":      0.20,
            "idle_storage_pct":  0.22,
            "reservation_waste": 0.05,
            "dev_leak_pct":      0.30,
        },
    },
}

AWS_INSTANCES = {
    "t3.micro":    {"vcpu": 2,  "ram": 1,   "price": 0.0104},
    "t3.small":    {"vcpu": 2,  "ram": 2,   "price": 0.0208},
    "t3.medium":   {"vcpu": 2,  "ram": 4,   "price": 0.0416},
    "t3.large":    {"vcpu": 2,  "ram": 8,   "price": 0.0832},
    "m5.large":    {"vcpu": 2,  "ram": 8,   "price": 0.096},
    "m5.xlarge":   {"vcpu": 4,  "ram": 16,  "price": 0.192},
    "m5.2xlarge":  {"vcpu": 8,  "ram": 32,  "price": 0.384},
    "c5.large":    {"vcpu": 2,  "ram": 4,   "price": 0.085},
    "c5.xlarge":   {"vcpu": 4,  "ram": 8,   "price": 0.170},
    "c5.2xlarge":  {"vcpu": 8,  "ram": 16,  "price": 0.340},
    "r5.large":    {"vcpu": 2,  "ram": 16,  "price": 0.126},
    "r5.xlarge":   {"vcpu": 4,  "ram": 32,  "price": 0.252},
    "p3.2xlarge":  {"vcpu": 8,  "ram": 61,  "price": 3.060},
    "p3.8xlarge":  {"vcpu": 32, "ram": 244, "price": 12.24},
}

AZURE_INSTANCES = {
    "Standard_B1s":    {"vcpu": 1,  "ram": 1,   "price": 0.0104},
    "Standard_B2s":    {"vcpu": 2,  "ram": 4,   "price": 0.0416},
    "Standard_D2s_v3": {"vcpu": 2,  "ram": 8,   "price": 0.096},
    "Standard_D4s_v3": {"vcpu": 4,  "ram": 16,  "price": 0.192},
    "Standard_D8s_v3": {"vcpu": 8,  "ram": 32,  "price": 0.384},
    "Standard_F4s":    {"vcpu": 4,  "ram": 8,   "price": 0.199},
    "Standard_E4s_v3": {"vcpu": 4,  "ram": 32,  "price": 0.252},
    "Standard_NC6":    {"vcpu": 6,  "ram": 56,  "price": 0.900},
}

AWS_REGIONS = {
    "ap-south-1":     {"name": "Mumbai",     "cost_multiplier": 1.00},
    "ap-southeast-1": {"name": "Singapore",  "cost_multiplier": 1.08},
    "us-east-1":      {"name": "N.Virginia", "cost_multiplier": 0.92},
    "eu-west-1":      {"name": "Ireland",    "cost_multiplier": 1.05},
}

AZURE_REGIONS = {
    "centralindia":   {"name": "Pune",       "cost_multiplier": 1.00},
    "southindia":     {"name": "Chennai",    "cost_multiplier": 1.00},
    "eastus":         {"name": "Virginia",   "cost_multiplier": 0.93},
    "westeurope":     {"name": "Amsterdam",  "cost_multiplier": 1.06},
}

SERVICES_AWS   = ["EC2", "S3", "RDS", "Lambda", "EKS", "ElastiCache", "CloudFront", "EBS"]
SERVICES_AZURE = ["VirtualMachines", "BlobStorage", "AzureSQL", "AKS", "Redis", "CDN", "ManagedDisks"]
ENVIRONMENTS   = ["production", "staging", "development", "testing", "qa"]
TEAMS          = ["backend", "frontend", "data", "ml", "devops", "security", "platform"]


def resource_id(company: str, idx: int) -> str:
    raw = f"res-{company}-{idx}-{SEED}"
    return f"res-{hashlib.md5(raw.encode()).hexdigest()[:10]}"

def jitter(base: float, pct: float = 0.08) -> float:
    return base * (1 + np.random.uniform(-pct, pct))

def apply_seasonality(date, base: float) -> float:
    m = 1.0
    if date.day >= 28:
        m *= np.random.uniform(1.15, 1.28)   # month-end batch spike
    if date.month in [1, 2, 3]:
        m *= 1.10                              # Q1 budget flush
    if date.weekday() >= 5:
        m *= np.random.uniform(0.70, 0.85)    # weekend dip
    return base * m

def build_resource_catalogue(company_key: str, profile: dict) -> pd.DataFrame:
    cloud  = profile["cloud"]
    size   = profile["size"]
    waste  = profile["waste_profile"]
    n_res  = 120 if size == "mid" else 200

    instances     = AWS_INSTANCES if cloud == "AWS" else AZURE_INSTANCES
    regions       = AWS_REGIONS   if cloud == "AWS" else AZURE_REGIONS
    services      = SERVICES_AWS  if cloud == "AWS" else SERVICES_AZURE
    instance_keys = list(instances.keys())
    region_keys   = list(regions.keys())

    rows = []
    for i in range(n_res):
        env     = np.random.choice(ENVIRONMENTS, p=[0.40, 0.20, 0.25, 0.10, 0.05])
        team    = np.random.choice(TEAMS)
        service = np.random.choice(services)
        itype   = np.random.choice(instance_keys)
        region  = np.random.choice(region_keys)

        # Determine waste label — probabilistic, not hand-labelled
        r, cum, label = random.random(), 0.0, "normal"
        is_storage = service in ["S3", "BlobStorage", "EBS", "ManagedDisks"]
        is_dev     = env in ["development", "staging", "testing", "qa"]
        patterns   = [
            ("zombie_instance",  waste["zombie_pct"]),
            ("overprovisioned",  waste["overprov_pct"]),
            ("idle_storage",     waste["idle_storage_pct"] if is_storage else 0),
            ("reservation_waste",waste["reservation_waste"]),
            ("dev_env_leak",     waste["dev_leak_pct"] if is_dev else 0),
        ]
        for lbl, prob in patterns:
            cum += prob
            if r < cum:
                label = lbl
                break

        is_reserved = (label == "reservation_waste") or (random.random() < 0.25 and env == "production")
        storage_gb  = round(np.random.lognormal(5, 1.5), 2) if is_storage else None

        rows.append({
            "resource_id":    resource_id(company_key, i),
            "company":        company_key,
            "account_id":     profile["account_id"],
            "cloud_provider": cloud,
            "service":        service,
            "instance_type":  itype,
            "region":         region,
            "environment":    env,
            "team":           team,
            "is_reserved":    is_reserved,
            "storage_gb":     storage_gb,
            "waste_label":    label,
            "created_at":     START_DATE + timedelta(days=random.randint(0, 180)),
        })
    return pd.DataFrame(rows)

def simulate_daily_metrics(resource: pd.Series, instances: dict, regions: dict, growth_rate: float) -> list:
    rows     = []
    inst     = instances.get(resource["instance_type"], list(instances.values())[0])
    reg      = regions.get(resource["region"],          list(regions.values())[0])
    label    = resource["waste_label"]
    env      = resource["environment"]
    created  = resource["created_at"]
    is_res   = resource["is_reserved"]

    base_price = inst["price"] * reg["cost_multiplier"]
    if is_res:
        base_price *= 0.60   # reserved discount

    for day_idx, date in enumerate(DATE_RANGE):
        if pd.Timestamp(date) < pd.Timestamp(created):
            continue

        growth_mult = 1 + (growth_rate * (day_idx / 30))
        is_weekend  = date.weekday() >= 5

        if label == "zombie_instance":
            cpu, mem = np.random.uniform(0.5, 4.5), np.random.uniform(5, 15)
            hours_on = 24.0
            get_req  = 0

        elif label == "overprovisioned":
            cpu, mem = np.random.uniform(5, 28), np.random.uniform(10, 38)
            hours_on = 24.0 if env == "production" else (0.0 if is_weekend else 12.0)
            get_req  = random.randint(100, 5000)

        elif label == "idle_storage":
            cpu, mem = 0.0, 0.0
            hours_on = 0.0
            get_req  = 0  

        elif label == "reservation_waste":
            cpu, mem = np.random.uniform(10, 38), np.random.uniform(10, 38)
            hours_on = np.random.uniform(2, 10)
            get_req  = random.randint(10, 500)

        elif label == "dev_env_leak":
            if is_weekend:
                cpu, mem = np.random.uniform(0.5, 8), np.random.uniform(5, 20)
                hours_on = 24.0   # running all weekend — the leak
                get_req  = random.randint(0, 50)
            else:
                cpu, mem = np.random.uniform(20, 70), np.random.uniform(20, 60)
                hours_on = np.random.uniform(8, 14)
                get_req  = random.randint(500, 8000)

        else: 
            cpu      = np.random.beta(3, 2) * 85 + 5
            mem      = np.random.beta(2, 2) * 70 + 15
            hours_on = 24.0 if env == "production" else (0.0 if is_weekend else np.random.uniform(6, 14))
            get_req  = random.randint(1000, 100000)

        raw_cost   = base_price * hours_on * growth_mult
        daily_cost = apply_seasonality(date, raw_cost)
        daily_cost = max(0.0, jitter(daily_cost, 0.05))

        if resource["service"] in ["S3", "BlobStorage", "EBS", "ManagedDisks"] and resource["storage_gb"]:
            daily_cost += resource["storage_gb"] * 0.023 / 30

        # composite waste score(0-1)
        idle_r     = max(0.0, 1 - cpu / 100)
        overprov_s = max(0.0, (1 - cpu / 100)) * max(0.0, (1 - mem / 100))
        zombie_sig = 1.0 if (cpu < 5 and hours_on >= 20) else 0.0
        waste_score = round(min(1.0, (0.4 * idle_r + 0.35 * overprov_s + 0.25 * zombie_sig) * jitter(1.0, 0.08)), 4)

        rows.append({
            "date":                 date.strftime("%Y-%m-%d"),
            "resource_id":          resource["resource_id"],
            "company":              resource["company"],
            "account_id":           resource["account_id"],
            "cloud_provider":       resource["cloud_provider"],
            "service":              resource["service"],
            "instance_type":        resource["instance_type"],
            "region":               resource["region"],
            "environment":          resource["environment"],
            "team":                 resource["team"],
            "is_reserved":          resource["is_reserved"],
            "storage_gb":           resource["storage_gb"],
            "usage_hours":          round(hours_on, 2),
            "cpu_utilisation":      round(cpu, 2),
            "memory_utilisation":   round(mem, 2),
            "get_requests":         int(get_req),
            "cost_usd":             round(daily_cost, 4),
            "waste_score":          waste_score,
            "waste_label":          label,
        })
    return rows

def generate_all():
    print("=" * 65)
    print("  CloudOptima AI — Synthetic Data Generator")
    print("  3 Indian SaaS Companies | AWS + Azure | 18 months")
    print("=" * 65)

    all_rows, catalogues = [], []

    for company_key, profile in COMPANIES.items():
        print(f"\n▶  {company_key.upper()} ({profile['city']}, {profile['cloud']})")

        instances = AWS_INSTANCES if profile["cloud"] == "AWS" else AZURE_INSTANCES
        regions   = AWS_REGIONS   if profile["cloud"] == "AWS" else AZURE_REGIONS

        catalogue = build_resource_catalogue(company_key, profile)
        catalogues.append(catalogue)
        print(f"   Resources: {len(catalogue)}")
        for lbl, cnt in catalogue["waste_label"].value_counts().items():
            pct = cnt / len(catalogue) * 100
            print(f"   {lbl:<22} {cnt:>3} resources  ({pct:.0f}%)")

        company_rows = []
        for _, resource in catalogue.iterrows():
            company_rows.extend(simulate_daily_metrics(resource, instances, regions, profile["growth_rate"]))
        print(f"   Daily rows: {len(company_rows):,}")
        all_rows.extend(company_rows)

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["company", "resource_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df_aws   = df[df["cloud_provider"] == "AWS"]
    df_azure = df[df["cloud_provider"] == "Azure"]

    df_aws.to_csv(  os.path.join(OUT_DIR, "billing_aws.csv"),      index=False)
    df_azure.to_csv(os.path.join(OUT_DIR, "billing_azure.csv"),    index=False)
    df.to_csv(      os.path.join(OUT_DIR, "billing_combined.csv"), index=False)

    cat_df = pd.concat(catalogues, ignore_index=True)
    cat_df.to_csv(  os.path.join(OUT_DIR, "resource_catalogue.csv"), index=False)

    total_waste = df[df["waste_label"] != "normal"]["cost_usd"].sum()

    print("\n" + "=" * 65)
    print("  GENERATION COMPLETE")
    print("=" * 65)
    print(f"  Total rows:            {len(df):,}")
    print(f"  Total resources:       {df['resource_id'].nunique()}")
    print(f"  Date range:            {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Total simulated spend: ${df['cost_usd'].sum():>12,.2f}")
    print(f"  Estimated waste spend: ${total_waste:>12,.2f}  ({total_waste/df['cost_usd'].sum()*100:.1f}%)")
    print(f"\n  Files → {OUT_DIR}/")
    print(f"    billing_aws.csv       ({len(df_aws):>8,} rows)")
    print(f"    billing_azure.csv     ({len(df_azure):>8,} rows)")
    print(f"    billing_combined.csv  ({len(df):>8,} rows)")
    print(f"    resource_catalogue.csv ({len(cat_df)} resources)")
    print("=" * 65)
    return df, cat_df


if __name__ == "__main__":
    generate_all()