"""Shared constants and preprocessing functions for Delhivery delay intelligence."""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]


def clean_shipments(raw: pd.DataFrame, medians: dict = None):
    """
    Takes a raw shipments DataFrame and returns a tuple (clean_df, medians).
    Steps in order:
      1. Drop exact duplicate rows.
      2. service_type: strip leading and trailing spaces and convert to title case.
      3. payment_mode: replace a missing value with the placeholder 'Unknown'.
      4. declared_value: remove rupee sign and commas, convert to numeric in declared_value_inr, drop text column.
      5. dispatch_time: convert to datetime.
      6. parcel_weight_kg: values > 100 treated as missing (scanner fault sentinel).
      7. fill missing distance_km, parcel_weight_kg, declared_value_inr with median.
    """
    df = raw.copy()
    
    # 1. Drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    
    # 2. service_type: strip leading/trailing spaces and title case
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()
    
    # 3. payment_mode: replace missing with 'Unknown'
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    
    # 4. declared_value: remove ₹ and commas, convert to float in declared_value_inr, drop declared_value
    clean_val = df["declared_value"].astype(str).str.replace("\u20b9", "", regex=False).str.replace(",", "", regex=False).str.strip()
    df["declared_value_inr"] = pd.to_numeric(clean_val, errors="coerce")
    df = df.drop(columns=["declared_value"])
    
    # 5. dispatch_time: convert to datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])
    
    # 6. parcel_weight_kg: values > 100 are scanner-fault sentinel -> treat as missing
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan
    
    # 7. fill missing distance_km, parcel_weight_kg, declared_value_inr with median
    if medians is None:
        medians = {
            "distance_km": float(df["distance_km"].median()),
            "parcel_weight_kg": float(df["parcel_weight_kg"].median()),
            "declared_value_inr": float(df["declared_value_inr"].median()),
        }
    else:
        medians = {k: float(v) for k, v in medians.items()}
        
    df["distance_km"] = df["distance_km"].fillna(medians["distance_km"])
    df["parcel_weight_kg"] = df["parcel_weight_kg"].fillna(medians["parcel_weight_kg"])
    df["declared_value_inr"] = df["declared_value_inr"].fillna(medians["declared_value_inr"])
    
    return df, medians


def build_features(clean: pd.DataFrame, hubs: pd.DataFrame) -> pd.DataFrame:
    """
    Attaches hub columns by a left merge on hub_id (hub_city, region, hub_tier, hub_daily_capacity)
    and constructs new feature columns:
      - promised_hours (from PROMISED_HOURS, integer)
      - is_cod (1 when payment_mode is COD, else 0 - Unknown gives 0)
      - is_weekend (1 when dispatch day is Saturday or Sunday)
      - dispatch_hour (hour of dispatch timestamp as integer)
      - log_declared_value (natural log of declared_value_inr)
    Does not use delivery_hours.
    """
    df = clean.copy()
    
    # Left merge with hubs table
    hub_cols = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    df = df.merge(hubs[hub_cols], on="hub_id", how="left")
    
    # Feature engineering
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"].astype(float))
    
    return df
