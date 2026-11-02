"""Package the trained model and associated artefacts for deployment."""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from src.shipment_prep import clean_shipments, build_features, FEATURES


def run_package_model():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Read config and raw data
    config_df = pd.read_csv("outputs/part_c_final_config.csv")
    config = config_df.iloc[0]

    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    clean, medians = clean_shipments(raw)
    df = build_features(clean, hubs)
    df["is_delayed"] = (df["delivery_hours"] > df["promised_hours"]).astype(int)

    X = df[FEATURES]
    y = df["is_delayed"]

    # Rebuild Part C's stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Build and fit final pipeline
    n_est = int(config["n_estimators"])
    m_depth = int(config["max_depth"]) if str(config["max_depth"]).isdigit() else None
    m_leaf = int(config["min_samples_leaf"])
    k_sel = int(config["selector_k"])
    threshold = float(config["threshold"])

    pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=k_sel)),
        ("model", RandomForestClassifier(n_estimators=n_est, max_depth=m_depth, min_samples_leaf=m_leaf, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    # Save artefacts in models/
    # 1. delay_model.pkl
    joblib.dump(pipeline, "models/delay_model.pkl")

    # 2. delay_threshold.csv
    pd.DataFrame([{"threshold": threshold}]).to_csv("models/delay_threshold.csv", index=False)

    # 3. feature_order.csv
    pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

    # 4. impute_values.csv
    impute_df = pd.DataFrame([
        {"column": k, "median": round(float(v), 4)} for k, v in medians.items()
    ])
    impute_df.to_csv("models/impute_values.csv", index=False)

    # Verification: reload with joblib.load and compare on test rows
    reloaded_pipeline = joblib.load("models/delay_model.pkl")

    in_mem_probs = pipeline.predict_proba(X_test)[:, 1]
    reloaded_probs = reloaded_pipeline.predict_proba(X_test)[:, 1]

    max_abs_diff = float(np.max(np.abs(in_mem_probs - reloaded_probs)))

    reload_check_df = pd.DataFrame([
        {"metric": "test_rows_scored", "value": len(X_test)},
        {"metric": "reload_max_abs_probability_diff", "value": round(max_abs_diff, 4)},
        {"metric": "threshold", "value": threshold},
    ])
    reload_check_df.to_csv("outputs/part_d_reload_check.csv", index=False)

    print(f"Packaging complete. Reload diff: {max_abs_diff:.6e}. Saved to models/ and outputs/part_d_reload_check.csv")


if __name__ == "__main__":
    run_package_model()
