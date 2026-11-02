"""Part A — Data Foundation and Delay Evidence."""
import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES, PROMISED_HOURS


def run_part_a():
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Read raw shipments and hubs
    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    raw_rows = len(raw)

    # Dedup check
    dedup = raw.drop_duplicates().reset_index(drop=True)
    duplicate_rows_removed = raw_rows - len(dedup)
    rows_after_dedup = len(dedup)

    # "Before" counts measured after removing duplicates and before any filling
    missing_distance_km_before = int(dedup["distance_km"].isna().sum())
    sentinel_weight_rows = int((dedup["parcel_weight_kg"] > 100).sum())
    # missing_parcel_weight_before does not include the sentinel rows
    missing_parcel_weight_before = int(dedup["parcel_weight_kg"].isna().sum())
    missing_declared_value_before = int(dedup["declared_value"].isna().sum())
    missing_payment_mode_before = int(dedup["payment_mode"].isna().sum())
    service_type_spellings_before = int(dedup["service_type"].nunique())

    # 2. Call clean_shipments
    clean, medians = clean_shipments(raw)
    service_type_spellings_after = int(clean["service_type"].nunique())

    median_distance_km = float(round(medians["distance_km"], 4))
    median_parcel_weight_kg = float(round(medians["parcel_weight_kg"], 4))
    median_declared_value_inr = float(round(medians["declared_value_inr"], 4))

    # 3. Call build_features
    model_df = build_features(clean, hubs)
    rows_after_hub_merge = len(model_df)

    # 4. Add target is_delayed
    model_df["is_delayed"] = (model_df["delivery_hours"] > model_df["promised_hours"]).astype(int)

    # Write modelling table
    model_df.to_csv("data/modelling_table.csv", index=False)

    # Metrics on modelling table
    cols_check = FEATURES + ["delivery_hours", "is_delayed"]
    nulls_in_features_and_target = int(model_df[cols_check].isna().sum().sum())
    delayed_count = int(model_df["is_delayed"].sum())
    delay_rate = float(round(model_df["is_delayed"].mean(), 4))

    # Delivery hours IQR and bounds
    q1 = float(model_df["delivery_hours"].quantile(0.25))
    q3 = float(model_df["delivery_hours"].quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    hours_outlier_count = int(((model_df["delivery_hours"] < lower_bound) | (model_df["delivery_hours"] > upper_bound)).sum())

    # Declared value metrics on median-filled declared_value_inr
    declared_value_mean_inr = float(round(model_df["declared_value_inr"].mean(), 4))
    declared_value_median_inr = float(round(model_df["declared_value_inr"].median(), 4))
    log_declared_value_mean = float(round(model_df["log_declared_value"].mean(), 4))
    log_declared_value_median = float(round(model_df["log_declared_value"].median(), 4))

    # Compile part_a_checks.csv
    checks = [
        ("raw_rows", raw_rows),
        ("duplicate_rows_removed", duplicate_rows_removed),
        ("rows_after_dedup", rows_after_dedup),
        ("missing_distance_km_before", missing_distance_km_before),
        ("missing_parcel_weight_before", missing_parcel_weight_before),
        ("sentinel_weight_rows", sentinel_weight_rows),
        ("missing_declared_value_before", missing_declared_value_before),
        ("missing_payment_mode_before", missing_payment_mode_before),
        ("service_type_spellings_before", service_type_spellings_before),
        ("service_type_spellings_after", service_type_spellings_after),
        ("median_distance_km", median_distance_km),
        ("median_parcel_weight_kg", median_parcel_weight_kg),
        ("median_declared_value_inr", median_declared_value_inr),
        ("rows_after_hub_merge", rows_after_hub_merge),
        ("nulls_in_features_and_target", nulls_in_features_and_target),
        ("delayed_count", delayed_count),
        ("delay_rate", delay_rate),
        ("hours_q1", round(q1, 4)),
        ("hours_q3", round(q3, 4)),
        ("hours_lower_bound", round(lower_bound, 4)),
        ("hours_upper_bound", round(upper_bound, 4)),
        ("hours_outlier_count", hours_outlier_count),
        ("declared_value_mean_inr", declared_value_mean_inr),
        ("declared_value_median_inr", declared_value_median_inr),
        ("log_declared_value_mean", log_declared_value_mean),
        ("log_declared_value_median", log_declared_value_median),
    ]
    pd.DataFrame(checks, columns=["metric", "value"]).to_csv("outputs/part_a_checks.csv", index=False)

    # 5. Delay evidence table: part_a_delay_by_group.csv
    bins = [0, 200, 400, 600, 2000]
    labels = ["0-200", "201-400", "401-600", "601+"]
    model_df["distance_band"] = pd.cut(model_df["distance_km"], bins=bins, labels=labels, right=True)

    group_rows = []
    for dim in ["service_type", "hub_tier", "distance_band"]:
        grp = model_df.groupby(dim, observed=False).agg(
            shipments=("shipment_id", "count"),
            delayed=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"),
            mean_hours=("delivery_hours", "mean"),
        ).reset_index().rename(columns={dim: "group"})
        grp["dimension"] = dim
        grp["delay_rate"] = grp["delay_rate"].round(4)
        grp["mean_hours"] = grp["mean_hours"].round(4)
        group_rows.append(grp[["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]])

    delay_by_group = pd.concat(group_rows, ignore_index=True)
    delay_by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)

    # 6. Correlations: part_a_correlations.csv
    num_cols = model_df.select_dtypes(include=["number"]).columns
    corr_matrix = model_df[num_cols].corr().round(4)
    corr_df = corr_matrix.reset_index().rename(columns={"index": "feature"})
    corr_df.to_csv("outputs/part_a_correlations.csv", index=False)

    print("Part A complete. Written data/modelling_table.csv and outputs/part_a_*.csv")


if __name__ == "__main__":
    run_part_a()
