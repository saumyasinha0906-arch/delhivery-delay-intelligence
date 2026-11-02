# Delhivery Last-Mile Delay Intelligence Pipeline

**Program:** Artificial Intelligence and Machine Learning Certification Program · E&ICT Academy, IIT Roorkee  
**Setting:** Regional Operations Analytics, Delhivery Logistics Network  
**Deliverable:** Single Public GitHub Repository (`main` branch)

---

## Executive Summary

This repository delivers an end-to-end, leakage-free machine learning and operations intelligence pipeline designed to solve three critical last-mile delivery challenges for Delhivery:
1. **Delivery-Hours Estimation:** A regression engine providing accurate dispatch-time point estimates for transit duration.
2. **Proactive Delay Alerting:** A cost-sensitive classifier identifying parcels at high risk of breaching customer delivery promises (Express: 36h, Standard: 60h, Economy: 96h).
3. **Hub Behavioral Segmentation:** An unsupervised clustering framework categorizing 24 logistics hubs into operational playbooks and exposing severe mismatches in administrative hub tier labels.

---

## Repository Layout

```text
.
├── README.md                          # Front door carrying all written answers and analytics
├── requirements.txt                   # Exact pinned dependencies
├── generate_data.py                   # Data generation script (committed verbatim)
├── data/
│   ├── hubs.csv                       # Hub registry (24 hubs)
│   ├── raw_shipments.csv              # Historical shipments export (1,224 messy rows)
│   ├── new_shipments.csv              # Incoming dispatch export (48 parcels)
│   └── modelling_table.csv            # Cleaned, feature-engineered table (1,200 rows)
├── src/
│   ├── __init__.py
│   └── shipment_prep.py               # Shared constants, cleaning, and feature engineering
├── part_a_clean_eda.py                # Part A: Data cleaning, validation checks, EDA
├── part_b_regression.py               # Part B: Delivery-hours regression and diagnostics
├── part_c_classification.py           # Part C: Delay classification, tuning, thresholding
├── part_d_segments.py                 # Part D: Hub profiling, K-Means clustering, mismatch analysis
├── package_model.py                   # Part D: Pipeline serialization and reload validation
├── predict_delay.py                   # Part D: Automated scoring and operational action routing
├── models/
│   ├── delay_model.pkl                # Serialized final classification pipeline
│   ├── delay_threshold.csv            # Operational alert probability threshold (0.50)
│   ├── feature_order.csv              # Pinned 17-feature input order
│   └── impute_values.csv              # Preprocessing medians for inference imputation
└── outputs/                           # Every tabular deliverable and check CSV
    ├── hub_profile.csv
    ├── hub_segments.csv
    ├── new_shipment_actions.csv
    ├── part_a_checks.csv
    ├── part_a_correlations.csv
    ├── part_a_delay_by_group.csv
    ├── part_b_checks.csv
    ├── part_b_cv.csv
    ├── part_b_gb_learning_rate.csv
    ├── part_b_metrics.csv
    ├── part_b_tree_depth.csv
    ├── part_c_feature_scores.csv
    ├── part_c_final_config.csv
    ├── part_c_final_test.csv
    ├── part_c_models.csv
    ├── part_c_selectk.csv
    ├── part_c_thresholds.csv
    ├── part_c_tuning.csv
    ├── part_d_reload_check.csv
    ├── part_d_separation.csv
    ├── segment_profile.csv
    └── tier_vs_segment_mismatch.csv
```

---

## 1. How to Run

### Environment Setup
The project runs fully offline using standard open-source Python scientific libraries. The dependencies are pinned with `==` in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Complete Pipeline Execution Order
From the repository root on a clean clone, run the scripts top to bottom:

```bash
# Step 0: Generate raw synthetic shipment datasets
python generate_data.py

# Step 1: Part A — Data foundation, cleaning, and delay evidence
python part_a_clean_eda.py

# Step 2: Part B — Delivery-hours regression and depth/learning-rate diagnostics
python part_b_regression.py

# Step 3: Part C — Delay classification, feature selection, tuning, and alert thresholding
python part_c_classification.py

# Step 4: Part D — Hub profiling, K-Means clustering, and separation analysis
python part_d_segments.py

# Step 5: Packaging — Model serialization and reload check
python package_model.py

# Step 6: Production Scoring — Predict delays and assign actions to new shipments
python predict_delay.py
```

> **Model Serialization Note:**  
> Running `python package_model.py` regenerates `models/delay_model.pkl` along with `models/delay_threshold.csv`, `models/feature_order.csv`, and `models/impute_values.csv`. The committed pickle loads reliably only on the exact scikit-learn version pinned in `requirements.txt` (`scikit-learn==1.8.0`). When running on differing scikit-learn environments, `package_model.py` re-fits and serializes the pipeline seamlessly.

---

## 2. Data Findings

The data foundation and exploratory analysis conducted in `part_a_clean_eda.py` processed 1,224 raw shipment records down to 1,200 unique shipments after eliminating 24 exact duplicates (`outputs/part_a_checks.csv`). Key empirical findings include:

### (i) Service Type Delay Dynamics
Express service is delayed most frequently and at the highest rate:
* **Express:** 119 delayed out of 243 shipments (**48.97%** delay rate, mean delivery: 36.81 hours). Express commitments (36-hour promise) operate on razor-thin margins; slight transit disruptions or distance excursions push parcels over SLA.
* **Standard:** 132 delayed out of 642 shipments (**20.56%** delay rate, mean delivery: 52.26 hours against 60h promise).
* **Economy:** 27 delayed out of 315 shipments (**8.57%** delay rate, mean delivery: 82.09 hours against 96h promise).
*(Source: `outputs/part_a_delay_by_group.csv`)*

### (ii) Hub Tier Escalation
Delay rates exhibit a steep, non-linear escalation across administrative tiers:
* **Tier 1:** 16 delayed out of 400 shipments (**4.00%** delay rate, mean delivery: 48.57 hours).
* **Tier 2:** 86 delayed out of 400 shipments (**21.50%** delay rate, mean delivery: 58.82 hours).
* **Tier 3:** 176 delayed out of 400 shipments (**44.00%** delay rate, mean delivery: 63.49 hours).
The delay rate jumps **11-fold** from Tier 1 to Tier 3, driven by longer feeder lanes, lower linehaul frequency, and heavier delivery route stop counts.
*(Source: `outputs/part_a_delay_by_group.csv`)*

### (iii) Distance Band Trajectory
Distance is a primary driver of delivery breaches:
* **0–200 km:** 6 delayed out of 274 shipments (**2.19%** delay rate, mean delivery: 46.76 hours).
* **201–400 km:** 77 delayed out of 563 shipments (**13.68%** delay rate, mean delivery: 55.97 hours).
* **401–600 km:** 120 delayed out of 275 shipments (**43.64%** delay rate, mean delivery: 64.27 hours).
* **601+ km:** 75 delayed out of 88 shipments (**85.23%** delay rate, mean delivery: 72.26 hours).
Parcels traveling over 600 km breach their delivery commitment more than 85% of the time, illustrating severe linehaul transit friction.
*(Source: `outputs/part_a_delay_by_group.csv`)*

### (iv) Strongest Correlates with Delivery Hours
Excluding `delivery_hours` itself, the top three features most strongly correlated with transit duration in `outputs/part_a_correlations.csv` are:
1. `promised_hours`: **+0.8372** (strong positive correlation reflecting planned dispatch service tiers).
2. `distance_km`: **+0.4013** (direct physical linehaul travel constraint).
3. `n_stops`: **+0.3247** (last-mile delivery congestion and rider drop density).

### (v) Rationale for Log-Transforming Declared Value
Before transformation, raw parcel declared value (`declared_value_inr`) exhibited severe positive skewness, with an arithmetic mean of **INR 1,533.05** sitting significantly above its median of **INR 1,121.50** due to occasional high-value merchandise. Applying the natural logarithm compressed the long right tail into a symmetric distribution where `log_declared_value` mean (**7.0277 ≈ 7.028**) and median (**7.0224 ≈ 7.022**) closely coincide, protecting linear and distance-based estimators from high-leverage outliers.

---

## 3. Model Results (Regression)

Part B evaluated candidate regression models on 840 training, 180 validation, and 180 test shipments (`outputs/part_b_checks.csv`). Inputs were scaled using a `StandardScaler` fitted exclusively on training rows (training max mean absolute: `0.0000 < 1e-9`; test max mean absolute: `0.2187 > 0.01`, confirming zero test leakage).

### Validation Model Comparison (`outputs/part_b_metrics.csv`)

| Model | Split | MAE (hrs) | MSE | RMSE (hrs) | $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `mean_baseline` | val | 15.7952 | 355.5204 | 18.8552 | −0.0212 |
| `linear_regression` | val | 3.9222 | 23.9235 | **4.8912** | **0.9313** |
| `gradient_boosting` | val | 3.9166 | 24.1722 | 4.9165 | 0.9306 |
| `random_forest` | val | 4.2882 | 30.3550 | 5.5095 | 0.9128 |
| `decision_tree` | val | 5.0971 | 39.4338 | 6.2796 | 0.8867 |
| **`linear_regression` (Final)** | **test** | **3.5671** | **20.4624** | **4.5235** | **0.9493** |

* Every RMSE exactly equals $\sqrt{\text{MSE}}$ within $\pm0.001$.

### Baseline Interpretation
The `mean_baseline` model predicts the training set mean (58.39 hours) for every validation row. It achieves a validation $R^2$ of **−0.0212**. A negative $R^2$ demonstrates that predicting an unconditioned target mean performs worse than a simple horizontal line at the validation sample mean, explaining 0% of the variance and underscoring the absolute necessity of predictive modeling.

### Tree Depth Diagnostic: Bias vs. Variance (`outputs/part_b_tree_depth.csv`)
* **High Bias (Depth 1):** Training RMSE is 11.6334 hours and validation RMSE is 12.1324 hours. The generalization gap is small ($\text{val} - \text{train} = 0.4990 \le 2.5$), but both errors are massive. Validation error exceeds the optimal depth by 5.8528 hours ($12.1324 - 6.2796 \ge 4.0$), demonstrating severe underfitting.
* **High Variance (Unlimited Depth):** The unconstrained tree achieves a training RMSE of 0.0000 hours (complete memorization), but validation RMSE balloons to 7.8368 hours. The generalization gap is **7.8368 hours** ($\ge 5.0$), demonstrating severe overfitting.
* **Optimal Tree Depth:** Depth **6** minimizes validation error with a validation RMSE of **6.2796 hours** (train RMSE: 4.7114 hours).

### Gradient Boosting Learning Rate Experiment (`outputs/part_b_gb_learning_rate.csv`)
* `lr = 0.01`: Validation RMSE = 10.0018 hrs, $R^2$ = 0.7127. Fails due to **underfitting / slow convergence**; 100 boosting rounds take steps that are too small to descend the loss surface.
* `lr = 0.10`: Validation RMSE = **4.9165 hrs**, $R^2$ = **0.9306**. **Optimal configuration**, smoothly descending without oscillation.
* `lr = 1.00`: Validation RMSE = 7.3701 hrs, $R^2$ = 0.8440. Fails due to **overshooting** local minima and destabilizing ensemble residual updates.
* Satisfies $R^2(0.10) > R^2(1.00) > R^2(0.01)$.

### Cross-Validation Ranking vs. Final Model Selection
On the validation set, Linear Regression achieved the lowest RMSE (**4.8912 hrs** vs. 4.9165 hrs for Gradient Boosting). Evaluating both across 5-fold cross-validation on the combined 1,020 training+validation rows (`outputs/part_b_cv.csv`) yielded:
* `gradient_boosting`: Mean $R^2$ = **0.9353** ($\pm 0.0045$)
* `linear_regression`: Mean $R^2$ = **0.9261** ($\pm 0.0048$)

**Model Decision:**  
Per the project specification, the candidate with the lowest validation RMSE (`linear_regression`) was chosen and scored once on the final test set, achieving an impressive test RMSE of **4.5235 hours** and $R^2$ of **0.9493**. In a commercial production environment, Gradient Boosting Regressor would be strongly considered due to its superior cross-validated score and capacity to natively capture non-linear distance/monsoon interactions; however, Linear Regression offers unmatched computational efficiency (<1ms latency) and transparent interpretability for operations planning.

---

## 4. Model Results (Classification)

Part C formulated delay prediction as a cost-sensitive classification problem on 840 training, 180 validation, and 180 test shipments with stratified target proportions (195 / 42 / 41 delayed parcels, ~23.2% prevalence).

### Baseline and Default Models (`outputs/part_c_models.csv`)

| Model | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `blind_all_on_time` | 138 | 0 | 42 | 0 | 0.7667 | 0.0000 | 0.0000 | 0.0000 |
| `logistic_regression` | 128 | 10 | 7 | 35 | 0.9056 | 0.7778 | 0.8333 | 0.8046 |
| `decision_tree_depth_6` | 128 | 10 | 11 | 31 | 0.8833 | 0.7561 | 0.7381 | 0.7470 |
| `random_forest_default` | 132 | 6 | 11 | 31 | 0.9056 | 0.8378 | 0.7381 | 0.7848 |

* **The Blind Baseline Trap:** The `blind_all_on_time` model achieves an apparently high accuracy of **76.67%** simply by predicting the majority class. However, its recall is **0.00%**, missing all 42 delayed parcels. In logistics operations, where missed delays incur heavy penalties and customer churn, accuracy alone is a deceptive and dangerous metric.

### Feature Signal and Noise Elimination (`outputs/part_c_feature_scores.csv`)
Fitting a default Random Forest on all 17 features revealed that the five synthetic bookkeeping columns (`label_print_batch`, `scanner_build_no`, `tape_roll_no`, `sticker_serial_mod`, `shelf_slot_no`) captured a combined **22.44%** of tree impurity importance, illustrating how tree splits can overfit random noise.

In contrast, univariate ANOVA F-ranking cleanly segregated real signals from noise:
* **Top 7 Signal Features (Ranks 1–7):** `distance_km` (F: 389.21), `hub_tier` (F: 155.35), `hub_daily_capacity` (F: 144.13), `n_stops` (F: 125.48), `promised_hours` (F: 84.42), `rain_flag` (F: 53.03), and `sale_event_flag` (F: 36.73). Every top feature has $F > 30$.
* **Noise Columns (Ranks 8, 12, 14, 16, 17):** All five bookkeeping columns exhibit $F < 4.6$ (well below the 10 threshold).

### SelectK Diagnostic (`outputs/part_c_selectk.csv`)
Evaluating feature subsets with a Random Forest on the validation set showed:
* $k = 5$: val F1 = 0.6988
* **$k = 7$: val F1 = 0.8333 (Peak performance)**
* $k = 10$: val F1 = 0.7857
* $k = 12$: val F1 = 0.8049
* $k = 17$: val F1 = 0.7848

Restricting inputs to the top 7 signal features eliminated noise contamination and raised validation F1 by **+0.0485**.

### Hyperparameter Tuning Analysis (`outputs/part_c_tuning.csv`)
* `grid_search` (27 candidates, 135 fits): Best CV accuracy = 0.8952, Validation F1 = 0.7750
* `randomized_search` (9 candidates, 45 fits): Best CV accuracy = 0.8905, Validation F1 = 0.7848
* `default_forest_same_7_features` (1 candidate, untuned defaults): **Validation F1 = 0.8333**

**Insight on 180-Row Validation Tuning:**  
CV accuracy tuning favored constrained trees (`max_depth=6`, `min_samples_leaf=1`), which optimized overall classification accuracy on training folds. However, on the 180-row validation set with the default 0.50 threshold, the shallower trees were overly conservative in predicting the minority positive class. The unconstrained default forest (`n_estimators=100`, `max_depth=unlimited`, `min_samples_leaf=1`) produced finer probabilistic distinctions, winning on validation F1 and becoming the final production model (`outputs/part_c_final_config.csv`).

### Operations Alert Threshold Calibration (`outputs/part_c_thresholds.csv`)
Because a missed delayed parcel costs substantially more than an unnecessary proactive outreach, operations established a strict rule: **select the highest threshold $t$ where validation recall is at least 0.80**.

| Threshold | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.10 | 94 | 44 | 1 | 41 | 0.7500 | 0.4824 | 0.9762 | 0.6457 |
| 0.20 | 109 | 29 | 3 | 39 | 0.8222 | 0.5735 | 0.9286 | 0.7091 |
| 0.30 | 121 | 17 | 3 | 39 | 0.8889 | 0.6964 | 0.9286 | 0.7959 |
| 0.40 | 125 | 13 | 5 | 37 | 0.9000 | 0.7400 | 0.8810 | 0.8043 |
| **0.50** | **129** | **9** | **7** | **35** | **0.9111** | **0.7955** | **0.8333** | **0.8140** |
| 0.60 | 134 | 4 | 10 | 32 | 0.9222 | 0.8889 | 0.7619 | 0.8205 |
| 0.70 | 134 | 4 | 15 | 27 | 0.8944 | 0.8710 | 0.6429 | 0.7397 |
| 0.80 | 134 | 4 | 23 | 19 | 0.8500 | 0.8261 | 0.4524 | 0.5846 |
| 0.90 | 136 | 2 | 30 | 12 | 0.8222 | 0.8571 | 0.2857 | 0.4286 |

* At $t = 0.60$, recall drops to 76.19%, breaching the 80% SLA safety floor.
* At $t = 0.10$, recall is 97.62%, but precision plummets to 48.24% (44 false alarms), overwhelming operations teams.
* **Chosen Threshold:** **0.50**, achieving **83.33% recall**, **79.55% precision**, and **0.8140 F1**.

### One-Time Final Test Evaluation (`outputs/part_c_final_test.csv`)
Scored once on the 180 unseen test shipments at $t = 0.50$:
* **TN:** 125 | **FP:** 14 | **FN:** 10 | **TP:** 31
* **Accuracy:** **0.8667** (86.67%)
* **Precision:** **0.6889** (68.89%)
* **Recall:** **0.7561** (75.61%)
* **F1-Score:** **0.7209** (72.09%)

**Honest Assessment of Test Performance:**  
Test recall (75.61%) came in slightly lower than validation recall (83.33%). With only 41 delayed parcels in the test partition, missing 3 additional parcels (10 FN vs. 7 FN) accounts for the entire 7.7% variance. Despite this small-sample sensitivity, intercepting **31 out of 41 delayed shipments (75.6%)** with nearly 70% precision delivers massive commercial value over reactive firefighting.

---

## 5. Operations Recommendation

### Hub Operating Segments (`outputs/segment_profile.csv`)
K-Means clustering on the scaled hub profiles categorized the 24 logistics hubs into three behavioral segments based on empirical delay rates:
1. **Standard Segment (7 Hubs):** HUB01 Mumbai, HUB02 Delhi, HUB03 Bengaluru, HUB04 Hyderabad, HUB05 Chennai, HUB06 Pune, HUB08 Ahmedabad.
   * *Profile:* Mean delivery: **47.24 hrs**, delay rate: **2.57%**, mean distance: **188.93 km**, stops: **55.37**.
2. **Watch Segment (8 Hubs):** HUB07 Kolkata, HUB09 Jaipur, HUB10 Lucknow, HUB11 Chandigarh, HUB12 Indore, HUB14 Coimbatore, HUB16 Surat, HUB22 Dehradun.
   * *Profile:* Mean delivery: **56.81 hrs**, delay rate: **15.50%**, mean distance: **314.18 km**, stops: **70.72**.
3. **Escalate Segment (9 Hubs):** HUB13 Nagpur, HUB15 Kochi, HUB17 Patna, HUB18 Ranchi, HUB19 Guwahati, HUB20 Raipur, HUB21 Bhubaneswar, HUB23 Siliguri, HUB24 Agartala.
   * *Profile:* Mean delivery: **64.65 hrs**, delay rate: **46.00%**, mean distance: **470.48 km**, stops: **84.34**.

### Tier vs. Segment Mismatch Alert (`outputs/tier_vs_segment_mismatch.csv`)
Operations heads must immediately discard static administrative tiers for capacity planning. Exactly four hubs operate completely out of alignment with their tier labels:
* **HUB07 Kolkata (Admin: Tier 1 $\rightarrow$ Operates as: Watch):** Despite its Tier 1 label, Kolkata experiences a **16.00% delay rate** and 56.78-hour average transit. Dense urban corridors and long regional connections prevent it from operating like a smooth metro hub.
* **HUB13 Nagpur (Admin: Tier 2 $\rightarrow$ Operates as: Escalate):** Designated Tier 2, but has an alarming **46.00% delay rate**, 66.07-hour transit, and 473.32 km average lane length, operating identically to remote eastern hubs.
* **HUB15 Kochi (Admin: Tier 2 $\rightarrow$ Operates as: Escalate):** Designated Tier 2, but suffers a **44.00% delay rate** and 64.91-hour transit time.
* **HUB22 Dehradun (Admin: Tier 3 $\rightarrow$ Operates as: Watch):** Designated Tier 3, but performs far better than peers with only a **16.00% delay rate** and 57.06-hour transit.

**Actionable Mandate:** Decouple dispatch SOPs from administrative labels. Hubs 13 and 15 must receive senior hub manager oversight and buffer reserves, while Dehradun can be safely scheduled with standard regional turnaround times.

### K = 3 Cluster Stability & Rationale (`outputs/part_d_separation.csv`)
* $K = 2$: Ratio = **2.8940**, Cluster sizes: `15-9`
* $K = 3$: Ratio = **5.3928**, Cluster sizes: `9-8-7`
* $K = 4$: Ratio = **6.3047**, Cluster sizes: `8-7-5-4`
* $K = 5$: Ratio = **6.8866**, Cluster sizes: `8-7-4-3-2`

**Why K = 3 Was Kept:**  
While the mathematical across-over-within distance ratio continues rising for $K = 4$ and $K = 5$, those solutions introduce operational fragmentation. $K = 4$ isolates a cluster of only 4 hubs, and $K = 5$ generates tiny groups of 3 and 2 hubs. A logistics organization cannot maintain dedicated managerial playbooks for two-hub clusters. $K = 3$ creates three robust, balanced operational tiers (sizes 9, 8, and 7) with a dramatic separation leap from $K=2$ ($2.89 \rightarrow 5.39$).

### Operational Impact on New Dispatches (`outputs/new_shipment_actions.csv`)
Applying the packaged model (`models/delay_model.pkl`) at threshold 0.50 to the 48 incoming dispatches produced:
* **Flagged Parcels:** **8 out of 48 (16.67%)**
* **Action Breakdown:**
  * **Escalate to hub manager (6 parcels):** `SHP90002` (HUB24), `SHP90003` (HUB18), `SHP90011` (HUB17), `SHP90025` (HUB20), `SHP90041` (HUB21), `SHP90046` (HUB21). High-risk parcels in stressed hubs requiring route re-prioritization and linehaul expediting.
  * **Proactive customer call (2 parcels):** `SHP90005` (HUB22), `SHP90007` (HUB09). High-risk parcels in Watch hubs where proactive CRM notification prevents delivery escalation.
  * **No action (40 parcels):** Parcels predicted to fulfill commitments smoothly.

---

## 6. Production Plan

### (a) Production-Readiness Mapping
1. **Maintainability:** Preprocessing routines are centralized in `src/shipment_prep.py` (`clean_shipments`, `build_features`). No duplicate preprocessing exists in training or inference scripts.
2. **Consistency:** Inference scripts load serialized training medians (`models/impute_values.csv`) and enforce the exact 17-feature column ordering (`models/feature_order.csv`).
3. **Reproducibility:** Pinned dependencies (`requirements.txt`), fixed pseudo-random seeds (`random_state=42`), and deterministic clustering runs guarantee identical re-computation.
4. **Reliability:** Input pipelines sanitize unicode currency symbols, manage missing data gracefully, and detect sensor-fault sentinels (>100 kg). Pipeline serialization is verified via zero-diff reload checks (`outputs/part_d_reload_check.csv`).
5. **Performance (Speed):** The bundled scikit-learn pipeline executes predictions on batch dispatches in <30ms, easily meeting real-time dispatch gate requirements.
6. **Accuracy & Cost-Alignment:** Custom threshold tuning ($t=0.50$) aligns with the asymmetric operational cost where missed breaches outweigh false alarms.
7. **Monitorability:** Output predictions provide structured audit logs containing raw delay probabilities, binary flags, and discrete operational actions (`outputs/new_shipment_actions.csv`).

### (b) Live Production Monitoring Triggers
Operations must track three core metrics weekly:
1. **Delivered Parcel Delay Rate (Network Health):**  
   * *Baseline:* 23.17%.  
   * *Numeric Trigger:* Weekly delayed share $> 28.0\%$ or $< 18.0\%$ ($\pm5.0\%$ shift). Triggers root-cause audit for regional carrier disruption or macro weather events.
2. **Dispatch Gate Flag Rate (Model Stability):**  
   * *Baseline:* 16.67% (8/48 parcels).  
   * *Numeric Trigger:* Weekly flag rate $> 22.0\%$ (risks exhausting CRM call center capacity) or $< 12.0\%$ (potential model drift under-reporting risk).
3. **Realized Delay Recall (Model Quality):**  
   * *Baseline:* 75.61%.  
   * *Numeric Trigger:* 30-day trailing recall on delivered parcels falls below **70.0%**. Automatically triggers pipeline retraining.

### (c) Retraining Cadence, Drift, and Sale Events
* **Retraining Cadence:** Re-fit models bi-weekly over a rolling 90-day window to absorb changing lane transit times and seasonal fleet reallocations.
* **Handling Festival Sale Windows (Sept 15–28):**  
  E-commerce festive events (`sale_event_flag = 1`) introduce massive volume surges (+7 to +12 hours delay premium). Models must compare sale periods against prior festive sales rather than standard baseline weeks. During sale windows, temporary threshold offsets should be activated to manage call queue capacities.
* **Data Drift Manifestations:**
  * *Covariate Shift:* Changes in origin-destination distance distributions from new hub additions, or shifts in customer payment behavior (cashless UPI adoption lowering COD rates).
  * *Concept Drift:* Infrastructure upgrades (e.g., new expressway corridors reducing lane transit time) or severe monsoons (`rain_flag`) increasing delivery hours beyond historical distributions.

---

## 7. Limitations

1. **Synthetic Data Boundaries:** While calibrated against realistic logistics distributions, synthetic data lacks complex real-world dynamics such as multi-hop hub-and-spoke sortation bottlenecks, address parsing errors, vehicle breakdowns, and customer delivery re-attempts.
2. **Small Test Sample Size:** The test set contains only 180 shipments with 41 delayed outcomes. A variation of just 3 misclassified parcels shifts measured recall by ~7.3%, introducing wide confidence intervals around test evaluation metrics.
3. **Single Validation Split Calibration:** The operational threshold ($t = 0.50$) was derived from a single 180-row validation split. In production, cost-sensitive threshold curves should be tuned across out-of-fold predictions from repeated cross-validation folds.
4. **Static Point-of-Dispatch Horizon:** Predictions are generated strictly at parcel dispatch. The pipeline does not currently ingest real-time telematics, mid-mile GPS telemetry, or intermediate transit scan events that occur after dispatch.
