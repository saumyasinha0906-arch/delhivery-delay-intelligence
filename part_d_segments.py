"""Part D — Hub Segments and Separation Analysis."""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform


def run_part_d():
    os.makedirs("outputs", exist_ok=True)

    # 1. Hub profile from modelling table
    df = pd.read_csv("data/modelling_table.csv")
    hubs = pd.read_csv("data/hubs.csv")

    hub_profile = df.groupby("hub_id", observed=False).agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    ).reset_index().sort_values("hub_id").reset_index(drop=True)

    # Round to 4 decimal places for CSV output
    hub_profile_out = hub_profile.copy()
    for col in ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]:
        hub_profile_out[col] = hub_profile_out[col].round(4)
    hub_profile_out.to_csv("outputs/hub_profile.csv", index=False)

    # 2. Scaling & Separation Analysis
    # Profile columns in ascending hub_id order
    cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    scaler = StandardScaler()
    X_hub_scaled = scaler.fit_transform(hub_profile[cols])

    # Pairwise Euclidean distances on scaled matrix (276 pairs)
    dists = squareform(pdist(X_hub_scaled, metric="euclidean"))
    n_hubs = len(hub_profile)

    sep_rows = []
    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_hub_scaled)
        counts = pd.Series(labels).value_counts().sort_values(ascending=False).values
        size_str = "-".join(map(str, counts))

        across_dists = []
        within_dists = []
        for i in range(n_hubs):
            for j in range(i + 1, n_hubs):
                d = dists[i, j]
                if labels[i] == labels[j]:
                    within_dists.append(d)
                else:
                    across_dists.append(d)

        ratio = float(np.mean(across_dists) / np.mean(within_dists))
        sep_rows.append({
            "K": k,
            "across_over_within": round(ratio, 4),
            "cluster_sizes": size_str
        })

    pd.DataFrame(sep_rows).to_csv("outputs/part_d_separation.csv", index=False)

    # 3. K = 3 Segments & Tier Mismatch
    km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    hub_profile["cluster"] = km3.fit_predict(X_hub_scaled)

    # Order clusters by mean delay_rate: Standard (lowest), Watch (middle), Escalate (highest)
    cluster_delay = hub_profile.groupby("cluster")["delay_rate"].mean().sort_values()
    cluster_to_seg = {c: name for c, name in zip(cluster_delay.index, ["Standard", "Watch", "Escalate"])}
    hub_profile["segment"] = hub_profile["cluster"].map(cluster_to_seg)

    # Merge hub metadata (hub_city, hub_tier)
    hub_profile = hub_profile.merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")

    # outputs/hub_segments.csv: hub_id, segment, hub_city, hub_tier
    hub_segments = hub_profile[["hub_id", "segment", "hub_city", "hub_tier"]].copy()
    hub_segments.to_csv("outputs/hub_segments.csv", index=False)

    # outputs/segment_profile.csv: segment, hubs, mean_delivery_hours, delay_rate, mean_distance_km, mean_n_stops
    # (means of the hub-level values)
    seg_profile = hub_profile.groupby("segment").agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean"),
    ).reindex(["Standard", "Watch", "Escalate"]).reset_index()

    for col in ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]:
        seg_profile[col] = seg_profile[col].round(4)
    seg_profile.to_csv("outputs/segment_profile.csv", index=False)

    # outputs/tier_vs_segment_mismatch.csv:
    # Expected: Tier 1 -> Standard, Tier 2 -> Watch, Tier 3 -> Escalate
    tier_map = {1: "Standard", 2: "Watch", 3: "Escalate"}
    hub_profile["expected_segment"] = hub_profile["hub_tier"].map(tier_map)
    mismatches = hub_profile[hub_profile["segment"] != hub_profile["expected_segment"]][["hub_id", "segment", "hub_city", "hub_tier"]]
    mismatches.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

    print("Part D complete. Written outputs/hub_*.csv, outputs/segment_*.csv, outputs/part_d_separation.csv")


if __name__ == "__main__":
    run_part_d()
