"""Predict delay for incoming new shipments and assign operational actions."""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import numpy as np
import pandas as pd
import joblib
from src.shipment_prep import clean_shipments, build_features


def run_predict_delay():
    os.makedirs("outputs", exist_ok=True)

    # 1. Load inputs and artefacts
    new_raw = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    hub_segments = pd.read_csv("outputs/hub_segments.csv")

    model = joblib.load("models/delay_model.pkl")
    threshold_val = float(pd.read_csv("models/delay_threshold.csv").iloc[0]["threshold"])
    feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
    impute_df = pd.read_csv("models/impute_values.csv")
    medians = dict(zip(impute_df["column"], impute_df["median"].astype(float)))

    # 2. Clean new shipments using stored medians
    clean_new, _ = clean_shipments(new_raw, medians=medians)

    # 3. Attach features
    df_features = build_features(clean_new, hubs)

    # 4. Arrange columns in feature_order
    X_new = df_features[feature_order]

    # 5. Predict delay probabilities
    probs = model.predict_proba(X_new)[:, 1]
    df_features["delay_probability"] = np.round(probs, 4)

    # 6. Flag based on threshold
    df_features["flagged"] = (df_features["delay_probability"] >= threshold_val).astype(int)

    # 7. Merge hub segment (hub_id, segment)
    # Ensure preserving file order of new_shipments
    df_scored = df_features.merge(hub_segments[["hub_id", "segment"]], on="hub_id", how="left")

    # 8. Assign operational actions
    def assign_action(row):
        if row["flagged"] == 1:
            if row["segment"] == "Escalate":
                return "Escalate to hub manager"
            else:
                return "Proactive customer call"
        return "No action"

    df_scored["action"] = df_scored.apply(assign_action, axis=1)

    # 9. Format output table in original file order
    output_cols = ["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]
    actions_df = df_scored[output_cols].copy()
    actions_df.to_csv("outputs/new_shipment_actions.csv", index=False)

    print("predict_delay complete. Written outputs/new_shipment_actions.csv")


if __name__ == "__main__":
    run_predict_delay()
