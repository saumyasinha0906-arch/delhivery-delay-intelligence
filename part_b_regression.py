"""Part B — Delivery-Hours Regression."""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES


def run_part_b():
    os.makedirs("outputs", exist_ok=True)

    # 1. Read modelling table
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["delivery_hours"]

    # Split into train (70%), val (15%), test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 2. Scaling without leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_scaled_max_abs_mean = float(np.max(np.abs(np.mean(X_train_scaled, axis=0))))
    test_scaled_max_abs_mean = float(np.max(np.abs(np.mean(X_test_scaled, axis=0))))

    checks_df = pd.DataFrame([
        ("train_rows", len(X_train)),
        ("val_rows", len(X_val)),
        ("test_rows", len(X_test)),
        ("train_scaled_max_abs_mean", round(train_scaled_max_abs_mean, 4)),
        ("test_scaled_max_abs_mean", round(test_scaled_max_abs_mean, 4)),
    ], columns=["metric", "value"])
    checks_df.to_csv("outputs/part_b_checks.csv", index=False)

    # 3. Baseline
    y_val_baseline = np.full_like(y_val, y_train.mean())
    base_mae = float(round(mean_absolute_error(y_val, y_val_baseline), 4))
    base_mse = float(round(mean_squared_error(y_val, y_val_baseline), 4))
    base_rmse = float(round(np.sqrt(base_mse), 4))
    base_r2 = float(round(r2_score(y_val, y_val_baseline), 4))

    # 4. Tree-depth diagnostic
    depths = [1, 2, 3, 4, 6, 8, 12, None]
    tree_records = []
    best_d = None
    best_d_val_rmse = float("inf")

    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        tr_rmse = float(round(np.sqrt(mean_squared_error(y_train, dt.predict(X_train))), 4))
        va_rmse = float(round(np.sqrt(mean_squared_error(y_val, dt.predict(X_val))), 4))
        d_label = str(d) if d is not None else "unlimited"
        tree_records.append({
            "max_depth": d_label,
            "train_rmse": tr_rmse,
            "val_rmse": va_rmse
        })
        if va_rmse < best_d_val_rmse:
            best_d_val_rmse = va_rmse
            best_d = d

    pd.DataFrame(tree_records).to_csv("outputs/part_b_tree_depth.csv", index=False)

    # 5. Four candidate models
    candidates = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": DecisionTreeRegressor(max_depth=best_d, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    metrics_rows = [
        {"model": "mean_baseline", "split": "val", "MAE": base_mae, "MSE": base_mse, "RMSE": base_rmse, "R2": base_r2}
    ]

    val_rmse_dict = {}
    fitted_models = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        preds = model.predict(X_val)
        mae = float(round(mean_absolute_error(y_val, preds), 4))
        mse = float(round(mean_squared_error(y_val, preds), 4))
        rmse = float(round(np.sqrt(mse), 4))
        r2 = float(round(r2_score(y_val, preds), 4))
        val_rmse_dict[name] = rmse
        metrics_rows.append({
            "model": name, "split": "val", "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2
        })

    # 6. Learning-rate experiment
    lr_records = []
    for lr in [0.01, 0.1, 1.0]:
        gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
        gbr.fit(X_train, y_train)
        preds = gbr.predict(X_val)
        va_rmse = float(round(np.sqrt(mean_squared_error(y_val, preds)), 4))
        va_r2 = float(round(r2_score(y_val, preds), 4))
        lr_records.append({
            "learning_rate": lr,
            "val_rmse": va_rmse,
            "val_r2": va_r2
        })
    pd.DataFrame(lr_records).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

    # 7. Cross-validated comparison and final model
    sorted_candidates = sorted(val_rmse_dict.keys(), key=lambda k: val_rmse_dict[k])
    top_two = sorted_candidates[:2]

    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_records = []
    for name in top_two:
        m = candidates[name]
        scores = cross_val_score(m, X_train_val, y_train_val, cv=kf)
        cv_records.append({
            "model": name,
            "mean_r2": float(round(scores.mean(), 4)),
            "std_r2": float(round(scores.std(), 4))
        })
    pd.DataFrame(cv_records).to_csv("outputs/part_b_cv.csv", index=False)

    # Final model is the candidate with lowest validation RMSE
    final_model_name = sorted_candidates[0]
    final_model = fitted_models[final_model_name]
    test_preds = final_model.predict(X_test)
    test_mae = float(round(mean_absolute_error(y_test, test_preds), 4))
    test_mse = float(round(mean_squared_error(y_test, test_preds), 4))
    test_rmse = float(round(np.sqrt(test_mse), 4))
    test_r2 = float(round(r2_score(y_test, test_preds), 4))

    metrics_rows.append({
        "model": final_model_name,
        "split": "test",
        "MAE": test_mae,
        "MSE": test_mse,
        "RMSE": test_rmse,
        "R2": test_r2
    })

    pd.DataFrame(metrics_rows).to_csv("outputs/part_b_metrics.csv", index=False)
    print("Part B complete. Written outputs/part_b_*.csv")


if __name__ == "__main__":
    run_part_b()
