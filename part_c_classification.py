"""Part C — Delay Classifier and Alert Threshold."""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS


def run_part_c():
    os.makedirs("outputs", exist_ok=True)

    # 1. Read modelling table
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]

    # Stratified split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # 2. Baseline and candidate models
    # Blind baseline: all on-time (predict 0)
    blind_preds = np.zeros_like(y_val)
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_val, blind_preds).ravel()
    acc_b = accuracy_score(y_val, blind_preds)

    # Logistic Regression
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
    lr_pipe.fit(X_train, y_train)
    lr_preds = lr_pipe.predict(X_val)
    tn_lr, fp_lr, fn_lr, tp_lr = confusion_matrix(y_val, lr_preds).ravel()
    acc_lr = accuracy_score(y_val, lr_preds)
    prec_lr = precision_score(y_val, lr_preds, zero_division=0)
    rec_lr = recall_score(y_val, lr_preds)
    f1_lr = f1_score(y_val, lr_preds)

    # Decision Tree diagnostic across depths
    best_dt_depth = None
    best_dt_f1 = -1
    best_dt_model = None
    for d in [2, 3, 4, 5, 6, 8, 10]:
        dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        dt_val_preds = dt.predict(X_val)
        val_f1 = f1_score(y_val, dt_val_preds)
        if val_f1 > best_dt_f1:
            best_dt_f1 = val_f1
            best_dt_depth = d
            best_dt_model = dt

    dt_preds = best_dt_model.predict(X_val)
    tn_dt, fp_dt, fn_dt, tp_dt = confusion_matrix(y_val, dt_preds).ravel()
    acc_dt = accuracy_score(y_val, dt_preds)
    prec_dt = precision_score(y_val, dt_preds, zero_division=0)
    rec_dt = recall_score(y_val, dt_preds)
    f1_dt = f1_score(y_val, dt_preds)

    # Default Random Forest
    rf_def = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_def.fit(X_train, y_train)
    rf_preds = rf_def.predict(X_val)
    tn_rf, fp_rf, fn_rf, tp_rf = confusion_matrix(y_val, rf_preds).ravel()
    acc_rf = accuracy_score(y_val, rf_preds)
    prec_rf = precision_score(y_val, rf_preds, zero_division=0)
    rec_rf = recall_score(y_val, rf_preds)
    f1_rf = f1_score(y_val, rf_preds)

    models_df = pd.DataFrame([
        {
            "model": "blind_all_on_time",
            "TN": int(tn_b), "FP": int(fp_b), "FN": int(fn_b), "TP": int(tp_b),
            "accuracy": round(acc_b, 4), "precision": 0.0, "recall": 0.0, "f1": 0.0
        },
        {
            "model": "logistic_regression",
            "TN": int(tn_lr), "FP": int(fp_lr), "FN": int(fn_lr), "TP": int(tp_lr),
            "accuracy": round(acc_lr, 4), "precision": round(prec_lr, 4), "recall": round(rec_lr, 4), "f1": round(f1_lr, 4)
        },
        {
            "model": f"decision_tree_depth_{best_dt_depth}",
            "TN": int(tn_dt), "FP": int(fp_dt), "FN": int(fn_dt), "TP": int(tp_dt),
            "accuracy": round(acc_dt, 4), "precision": round(prec_dt, 4), "recall": round(rec_dt, 4), "f1": round(f1_dt, 4)
        },
        {
            "model": "random_forest_default",
            "TN": int(tn_rf), "FP": int(fp_rf), "FN": int(fn_rf), "TP": int(tp_rf),
            "accuracy": round(acc_rf, 4), "precision": round(prec_rf, 4), "recall": round(rec_rf, 4), "f1": round(f1_rf, 4)
        },
    ])
    models_df.to_csv("outputs/part_c_models.csv", index=False)

    # 3. Which columns carry signal?
    rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_full.fit(X_train, y_train)
    rf_importances = rf_full.feature_importances_

    kbest_all = SelectKBest(f_classif, k="all")
    kbest_all.fit(X_train, y_train)
    f_scores = kbest_all.scores_

    feature_df = pd.DataFrame({
        "feature": FEATURES,
        "f_score": [round(float(s), 4) for s in f_scores],
        "rf_importance": [round(float(imp), 4) for imp in rf_importances],
        "is_noise": [f in NOISE_COLUMNS for f in FEATURES],
    })
    feature_df["f_rank"] = feature_df["f_score"].rank(ascending=False, method="min").astype(int)
    feature_df = feature_df.sort_values("f_rank").reset_index(drop=True)
    feature_df[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].to_csv("outputs/part_c_feature_scores.csv", index=False)

    # SelectK experiments
    selectk_rows = []
    for k in [5, 7, 10, 12, 17]:
        pipe_k = Pipeline([
            ("sel", SelectKBest(f_classif, k=k)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pipe_k.fit(X_train, y_train)
        pred_k = pipe_k.predict(X_val)
        selectk_rows.append({"k": k, "val_f1": round(float(f1_score(y_val, pred_k)), 4)})
    pd.DataFrame(selectk_rows).to_csv("outputs/part_c_selectk.csv", index=False)

    # 4. Tuning on top 7 features
    kbest_7 = SelectKBest(f_classif, k=7)
    X_train_7 = kbest_7.fit_transform(X_train, y_train)
    X_val_7 = kbest_7.transform(X_val)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10]
    }

    gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    gs.fit(X_train_7, y_train)
    gs_val_preds = gs.predict(X_val_7)
    gs_val_f1 = float(round(f1_score(y_val, gs_val_preds), 4))
    gs_best_acc = float(round(gs.best_score_, 4))

    rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=9, cv=5, random_state=42)
    rs.fit(X_train_7, y_train)
    rs_val_preds = rs.predict(X_val_7)
    rs_val_f1 = float(round(f1_score(y_val, rs_val_preds), 4))
    rs_best_acc = float(round(rs.best_score_, 4))

    # Untuned forest on same 7 features: n_estimators=100, no depth limit, min_samples_leaf=1
    def_rf_7 = RandomForestClassifier(n_estimators=100, random_state=42)
    def_rf_7.fit(X_train_7, y_train)
    def_rf_preds = def_rf_7.predict(X_val_7)
    def_rf_val_f1 = float(round(f1_score(y_val, def_rf_preds), 4))

    tuning_df = pd.DataFrame([
        {
            "search": "grid_search",
            "candidates": 27,
            "cv_folds": 5,
            "total_fits": 135,
            "best_params": str(gs.best_params_),
            "best_cv_accuracy": gs_best_acc,
            "val_f1": gs_val_f1
        },
        {
            "search": "randomized_search",
            "candidates": 9,
            "cv_folds": 5,
            "total_fits": 45,
            "best_params": str(rs.best_params_),
            "best_cv_accuracy": rs_best_acc,
            "val_f1": rs_val_f1
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "defaults",
            "best_cv_accuracy": "",
            "val_f1": def_rf_val_f1
        },
    ])
    tuning_df.to_csv("outputs/part_c_tuning.csv", index=False)

    # 5. Final model selection
    # Ties -> default_forest_same_7_features, then grid_search, then randomized_search
    # Check max val_f1
    candidates_scores = [
        ("default_forest_same_7_features", def_rf_val_f1, {"n_estimators": 100, "max_depth": "unlimited", "min_samples_leaf": 1}),
        ("grid_search", gs_val_f1, gs.best_params_),
        ("randomized_search", rs_val_f1, rs.best_params_)
    ]
    candidates_scores.sort(key=lambda x: x[1], reverse=True)
    winner_name, winner_f1, winner_params = candidates_scores[0]

    if winner_name == "default_forest_same_7_features":
        chosen_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        n_est = 100
        m_depth = "unlimited"
        m_leaf = 1
    elif winner_name == "grid_search":
        chosen_rf = RandomForestClassifier(**gs.best_params_, random_state=42)
        n_est = gs.best_params_["n_estimators"]
        m_depth = gs.best_params_["max_depth"]
        m_leaf = gs.best_params_["min_samples_leaf"]
    else:
        chosen_rf = RandomForestClassifier(**rs.best_params_, random_state=42)
        n_est = rs.best_params_["n_estimators"]
        m_depth = rs.best_params_["max_depth"]
        m_leaf = rs.best_params_["min_samples_leaf"]

    final_pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", chosen_rf)
    ])
    final_pipeline.fit(X_train, y_train)

    # 6. Alert threshold
    val_probs = final_pipeline.predict_proba(X_val)[:, 1]
    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresh_records = []

    for t in threshold_list:
        p_binary = (val_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, p_binary).ravel()
        acc = accuracy_score(y_val, p_binary)
        prec = precision_score(y_val, p_binary, zero_division=0)
        rec = recall_score(y_val, p_binary, zero_division=0)
        f1_t = f1_score(y_val, p_binary, zero_division=0)
        thresh_records.append({
            "threshold": t,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1_t, 4)
        })

    thresh_df = pd.DataFrame(thresh_records)
    thresh_df.to_csv("outputs/part_c_thresholds.csv", index=False)

    # Operations rule: highest t whose validation recall is at least 0.80 (if none, 0.1)
    eligible = thresh_df[thresh_df["recall"] >= 0.80]
    chosen_threshold = float(eligible["threshold"].max()) if len(eligible) > 0 else 0.1

    # 7. One test evaluation
    test_probs = final_pipeline.predict_proba(X_test)[:, 1]
    test_binary = (test_probs >= chosen_threshold).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, test_binary).ravel()
    acc_t = accuracy_score(y_test, test_binary)
    prec_t = precision_score(y_test, test_binary, zero_division=0)
    rec_t = recall_score(y_test, test_binary, zero_division=0)
    f1_test = f1_score(y_test, test_binary, zero_division=0)

    final_test_df = pd.DataFrame([{
        "threshold": chosen_threshold,
        "TN": int(tn_t), "FP": int(fp_t), "FN": int(fn_t), "TP": int(tp_t),
        "accuracy": round(acc_t, 4), "precision": round(prec_t, 4),
        "recall": round(rec_t, 4), "f1": round(f1_test, 4)
    }])
    final_test_df.to_csv("outputs/part_c_final_test.csv", index=False)

    final_config_df = pd.DataFrame([{
        "selector_k": 7,
        "n_estimators": n_est,
        "max_depth": m_depth,
        "min_samples_leaf": m_leaf,
        "threshold": chosen_threshold,
        "chosen_from": winner_name
    }])
    final_config_df.to_csv("outputs/part_c_final_config.csv", index=False)

    print("Part C complete. Written outputs/part_c_*.csv")


if __name__ == "__main__":
    run_part_c()
