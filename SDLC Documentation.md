
---

# Insurance Cost / Claim Severity Prediction

## Software Development Life Cycle, Technical Design & Model Governance Document

**Document ID:** INS-DS-008
**Project:** Insurance Cost / Claim Severity Prediction
**Project Series:** 100-Day Industry Data Science Project — Day 8
**Domain:** Health Insurance / Actuarial Analytics / Underwriting
**Document Type:** SDLC + Technical Design + Model Validation + Operations Handover
**Model Type:** Supervised Regression
**Final Algorithm:** XGBoost Regressor
**Dataset:** Medical Cost Personal Datasets
**Status:** Proof-of-Concept / Production-Architecture Demonstration
**Version:** 1.0
**Date:** September 2026

---

# Document Control

| Attribute           | Details                                                          |
| ------------------- | ---------------------------------------------------------------- |
| Document ID         | INS-DS-008                                                       |
| Version             | 1.0                                                              |
| Model               | Insurance Cost / Claim Severity Prediction                       |
| Business Domain     | Health Insurance                                                 |
| Primary Consumers   | Actuarial, Underwriting, Pricing, Risk, Data Science             |
| Technical Consumers | ML Engineering, Software Engineering, MLOps                      |
| Lifecycle           | Development → Validation → Deployment → Monitoring → Maintenance |
| Dataset             | Medical Cost Personal Datasets                                   |
| Records             | 1,338 raw / 1,337 after deduplication                            |
| Final Model         | XGBoost Regressor                                                |
| Model Features      | 15                                                               |
| Test Population     | 201 individuals                                                  |
| Deployment          | FastAPI                                                          |
| Monitoring          | PSI + performance threshold                                      |
| Production Status   | POC; not production-approved                                     |

---

# 1. Executive Summary

## 1.1 Business Objective

The objective of this project is to estimate the expected annual medical insurance cost of an individual policyholder using demographic and lifestyle characteristics.

The intended business application is to provide an analytical foundation for:

* risk-based pricing,
* reserve planning,
* underwriting support,
* portfolio segmentation,
* policyholder-level cost estimation.

The project evaluates whether individualized cost estimates can reduce aggregate reserving error compared with assigning every policyholder a single historical-average cost.

---

## 1.2 Business Problem

A flat-rate pricing or reserving approach assigns the same expected cost to individuals despite meaningful differences in risk characteristics.

The project therefore evaluates:

> **Whether an individualized machine-learning estimate can more accurately approximate actual annual medical cost than a flat historical-average estimate.**

The target variable, `charges`, is an actual dollar-valued annual medical insurance cost in the source dataset.

Consequently, the project is able to calculate a directly observed dollar-valued comparison between:

1. a flat-rate baseline, and
2. individualized model predictions.

No externally assumed claim amount, retention rate, replacement-cost multiplier, or other financial conversion factor is required for this comparison.

---

# 2. Scope

## 2.1 In Scope

The project covers:

* raw data ingestion,
* data-quality assessment,
* duplicate detection,
* exploratory data analysis,
* target-distribution analysis,
* categorical encoding,
* BMI categorization,
* interaction engineering,
* target transformation,
* stratified dataset splitting,
* baseline model development,
* candidate model comparison,
* final model selection,
* held-out test evaluation,
* business validation,
* model serialization,
* REST API deployment,
* API testing,
* feature-drift monitoring,
* performance monitoring,
* retraining-trigger simulation,
* technical documentation,
* model limitations.

---

## 2.2 Out of Scope

The current implementation does **not** constitute a production underwriting system.

The following are outside the current scope:

* medical diagnosis,
* medical-claims adjudication,
* fraud detection,
* real actuarial pricing certification,
* regulatory approval,
* production policy issuance,
* medical-history modeling,
* longitudinal claims modeling,
* healthcare-cost inflation modeling,
* real-time claims ingestion,
* production identity/access management,
* enterprise model governance,
* production-grade fairness certification.

---

# 3. Stakeholder & Responsibility Model

A production implementation would involve multiple stakeholders.

| Stakeholder        | Responsibility                                                |
| ------------------ | ------------------------------------------------------------- |
| Actuarial Team     | Cost assumptions, reserving methodology, actuarial validation |
| Underwriting       | Risk interpretation and business applicability                |
| Pricing Team       | Pricing strategy and policy-level application                 |
| Data Science       | Feature engineering, modeling, evaluation                     |
| ML Engineering     | Packaging, deployment, inference reliability                  |
| MLOps              | CI/CD, model registry, monitoring, retraining                 |
| Data Engineering   | Data pipelines and data-quality controls                      |
| Model Risk         | Independent model validation and governance                   |
| Compliance / Legal | Regulatory and usage review                                   |
| Business Owner     | Acceptance criteria and business sign-off                     |

For this project, the implementation demonstrates the technical workflow; formal production sign-off is not claimed.

---

# 4. SDLC Phase 1 — Business Problem Definition

## 4.1 Problem Statement

The insurer requires a method for estimating expected annual medical cost at policyholder level.

The solution should capture nonlinear relationships and interactions between risk characteristics rather than relying exclusively on a simple additive model.

---

## 4.2 Business Question

> How accurately can annual medical cost be estimated from the available policyholder characteristics, and how much aggregate reserving error can be reduced compared with a flat historical-average estimate?

---

## 4.3 Business Objective

The project has two objectives.

### Primary Technical Objective

Minimize prediction error.

### Primary Business Objective

Increase the proportion of policyholders whose estimated annual cost falls within a practically useful range of the actual cost.

The principal business metric used for model selection is:

**Percentage of predictions within ±20% of actual charges.**

---

## 4.4 Success Criteria

The solution must:

* outperform the simple baseline,
* provide measurable predictive accuracy,
* provide a reproducible inference pipeline,
* expose the model through an API,
* produce monitoring outputs,
* document known limitations,
* quantify business impact using held-out observations.

---

# 5. SDLC Phase 2 — Requirements Analysis

## 5.1 Functional Requirements

### FR-01 — Data Ingestion

The system shall load the insurance dataset from the configured data location.

### FR-02 — Data Validation

The system shall validate:

* expected columns,
* data types,
* missing values,
* duplicate records,
* target availability.

### FR-03 — Feature Engineering

The system shall generate:

* BMI category,
* smoker × BMI interaction,
* smoker-and-obese indicator,
* encoded categorical variables.

### FR-04 — Prediction

The system shall generate an estimated annual medical cost.

### FR-05 — API

The system shall expose the prediction through a REST API.

### FR-06 — Health Monitoring

The service shall expose a health endpoint.

### FR-07 — Model Monitoring

The system shall calculate feature drift and performance deterioration.

### FR-08 — Retraining Trigger

The monitoring process shall recommend retraining when the defined performance threshold is exceeded.

---

# 6. SDLC Phase 3 — Feasibility Analysis

## 6.1 Data Feasibility

The source dataset contains:

* 1,338 observations,
* six predictor variables,
* one continuous target.

The available variables are sufficient to demonstrate a supervised regression workflow.

---

## 6.2 Technical Feasibility

The problem can be modeled using:

* Linear Regression,
* Random Forest,
* XGBoost.

A FastAPI-based inference service provides a lightweight deployment mechanism.

---

## 6.3 Business Feasibility

Because the target is an actual dollar amount, the model can be evaluated using real observed costs rather than an assumed financial conversion.

However, the dataset is not sufficiently rich to establish production actuarial validity.

---

# 7. SDLC Phase 4 — Project Planning

## 7.1 Delivery Pipeline

```text
Business Requirement
        ↓
Data Acquisition
        ↓
Data Quality
        ↓
Exploratory Analysis
        ↓
Feature Engineering
        ↓
Dataset Splitting
        ↓
Baseline
        ↓
Candidate Models
        ↓
Model Selection
        ↓
Held-Out Evaluation
        ↓
Business Validation
        ↓
Model Packaging
        ↓
API Deployment
        ↓
Testing
        ↓
Monitoring
        ↓
Maintenance / Retraining
```

---

## 7.2 Project Deliverables

| Deliverable                | Purpose                   |
| -------------------------- | ------------------------- |
| `data_loader.py`           | Data ingestion            |
| `eda.py`                   | Exploratory analysis      |
| `clean_and_engineer.py`    | Data preparation          |
| `split.py`                 | Dataset splitting         |
| `train_model.py`           | Training and evaluation   |
| `app.py`                   | API service               |
| `monitor.py`               | Monitoring                |
| `requirements.txt`         | Environment dependencies  |
| `README.md`                | Operational setup         |
| `PROJECT_DOCUMENTATION.md` | Technical handover        |
| `model_card.json`          | Model governance artifact |
| `experiment_log.csv`       | Experiment tracking       |
| `business_validation.json` | Business validation       |
| `feature_importance.csv`   | Explainability artifact   |

---

# 8. SDLC Phase 5 — Data Acquisition & Data Quality

## 8.1 Source

**Medical Cost Personal Datasets**

The dataset contains annual insurance charges alongside:

* age,
* sex,
* BMI,
* children,
* smoker status,
* region.

---

## 8.2 Raw Data Profile

| Attribute                  |     Value |
| -------------------------- | --------: |
| Raw records                |     1,338 |
| Predictor fields           |         6 |
| Target                     | `charges` |
| Duplicate records          |         1 |
| Post-deduplication records |     1,337 |

---

## 8.3 Duplicate Handling

One exact duplicate was identified.

The duplicate matched across all available fields, including the target.

It was therefore removed because retaining it would artificially increase the representation of one observation.

---

## 8.4 Target Distribution

The raw target had a skewness of:

**1.516**

The distribution is strongly right-skewed due in part to the high-cost smoker segment.

The model training process therefore uses:

```text
log1p(charges)
```

to reduce the effect of extreme target values during training.

---

# 9. SDLC Phase 6 — Exploratory Data Analysis

EDA was performed before model development to identify structural relationships that could inform feature engineering.

---

## 9.1 Smoking Effect

| Segment    | Mean Annual Charges |
| ---------- | ------------------: |
| Non-smoker |              $8,434 |
| Smoker     |             $32,050 |

This represents approximately a **3.8× difference**.

Smoking was consequently identified as the strongest individual cost driver.

---

## 9.2 Regional Effect

| Region    | Mean Annual Charges |
| --------- | ------------------: |
| Southwest |             $12,347 |
| Southeast |             $14,735 |

The observed regional variation was comparatively modest.

---

## 9.3 BMI Effect

| BMI Category | Mean Annual Charges |
| ------------ | ------------------: |
| Underweight  |              $8,658 |
| Obese        |             $15,561 |

BMI showed an observable relationship with cost, but its standalone effect was substantially smaller than smoking.

---

## 9.4 Interaction Analysis

The most important EDA finding was the interaction between smoking status and BMI.

| Segment           | Mean Annual Charges |
| ----------------- | ------------------: |
| Non-smoking obese |              $8,853 |
| Smoking obese     |             $41,693 |

This represents approximately a **4.7× difference**.

The finding provided the empirical rationale for explicitly engineering:

```text
smoker_bmi_interaction
```

---

# 10. SDLC Phase 7 — Data Preparation & Feature Engineering

## 10.1 Feature Transformation

The raw variables were transformed into model-ready features.

| Feature                  | Transformation               |
| ------------------------ | ---------------------------- |
| `age`                    | Numeric                      |
| `sex`                    | Binary encoding              |
| `bmi`                    | Numeric                      |
| `children`               | Numeric                      |
| `smoker`                 | Binary encoding              |
| `region`                 | One-hot encoding             |
| `bmi_category`           | Clinical category + encoding |
| `smoker_bmi_interaction` | `smoker × bmi`               |
| `is_smoker_and_obese`    | Binary indicator             |

---

## 10.2 BMI Categorization

BMI categories were based on established clinical thresholds rather than bins optimized against this dataset.

This design improves conceptual transferability to a future underwriting environment.

---

## 10.3 Interaction Feature

The feature:

```text
smoker_bmi_interaction = smoker × bmi
```

was introduced specifically because EDA demonstrated that the relationship between smoking and BMI was not adequately represented by considering the variables independently.

---

## 10.4 Final Feature Set

The final model contains:

**15 engineered features.**

---

# 11. SDLC Phase 8 — Model Development

Three modeling levels were evaluated.

## 11.1 Baseline Model

**Algorithm:** Linear Regression

**Features:**

* age
* smoker

Validation:

| Metric      |    Result |
| ----------- | --------: |
| MAE         | $5,133.63 |
| Within ±20% |    56.72% |

The baseline establishes the minimum performance expected from a simple model.

---

## 11.2 Candidate Model — Random Forest

| Metric         |        Result |
| -------------- | ------------: |
| Validation MAE | **$1,951.57** |
| Within ±20%    |        74.13% |

Random Forest produced a substantial improvement over the baseline.

---

## 11.3 Candidate Model — XGBoost

| Metric         |     Result |
| -------------- | ---------: |
| Validation MAE |  $2,212.86 |
| Within ±20%    | **74.63%** |

---

# 12. SDLC Phase 9 — Model Evaluation & Selection

## 12.1 Model Comparison

| Model             | Validation MAE | Within ±20% |
| ----------------- | -------------: | ----------: |
| Linear Regression |      $5,133.63 |      56.72% |
| Random Forest     |  **$1,951.57** |      74.13% |
| XGBoost           |      $2,212.86 |  **74.63%** |

---

## 12.2 Selection Rationale

Random Forest achieved the lowest validation MAE.

XGBoost achieved the highest validation percentage within ±20%.

Because the business criterion emphasized the proportion of individual estimates falling within a predefined actionable range, XGBoost was selected.

This is a deliberate **metric-priority decision**, not a claim that XGBoost minimizes every error metric.

---

# 13. Held-Out Test Evaluation

The final model was evaluated on a separate held-out test set containing:

**201 individuals.**

| Metric            |        Result |
| ----------------- | ------------: |
| MAE               | **$2,338.17** |
| MAPE              |    **19.35%** |
| R²                |    **0.8584** |
| Within ±20%       |    **69.65%** |
| Test observations |       **201** |

The held-out test results are reported separately from validation results to preserve the distinction between **model selection** and **final evaluation**.

---

# 14. Business Validation

This project includes a directly measured business comparison.

## 14.1 Flat-Rate Baseline

Every test individual receives the training-set average:

**$13,352.08**

Aggregate absolute reserving error:

**$1,826,446.39**

---

## 14.2 Individualized Model

Aggregate absolute error from model predictions:

**$469,971.31**

---

## 14.3 Measured Error Reduction

$$
1,826,446.39 - 469,971.31
$$

=

**$1,356,475.07**

Relative reduction:

**74.27%**

---

## 14.4 Interpretation

The result means that, **within this held-out dataset**, individualized model predictions produced substantially lower aggregate absolute error than assigning the same training-average cost to every test individual.

It does **not** establish that a production insurer would achieve a 74.27% reduction in real-world reserves.

The observed result is specific to:

* this dataset,
* this sample,
* this split,
* this target definition,
* this modeling pipeline.

A production business case would require validation on representative historical claims/policy data.

---

# 15. SDLC Phase 10 — Deployment

## 15.1 Deployment Architecture

```text
                   ┌──────────────────┐
                   │ Policyholder Data│
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ FastAPI Request  │
                   └────────┬─────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │ Feature Transformation   │
              │ - Encoding              │
              │ - BMI category          │
              │ - Interaction           │
              └────────────┬─────────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ XGBoost Model     │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Annual Cost       │
                 │ Estimate          │
                 └───────────────────┘
```

---

## 15.2 API

### `GET /`

Browser-based prediction interface.

### `GET /health`

Service health check.

Expected response:

```json
{
  "status": "ok"
}
```

### `GET /docs`

Swagger/OpenAPI interface.

### `POST /estimate`

Generates an annual cost estimate.

---

## 15.3 Example

Input:

```json
{
  "age": 52,
  "sex": "male",
  "bmi": 34.5,
  "children": 1,
  "smoker": "yes",
  "region": "southeast"
}
```

Observed model output:

```json
{
  "estimated_annual_cost": 46658.92,
  "top_factors": [
    "smoker",
    "smoker_bmi_interaction",
    "age"
  ]
}
```

---

# 16. SDLC Phase 11 — Testing & Quality Assurance

## 16.1 Data Tests

Validated:

* schema,
* record count,
* duplicate handling,
* feature availability,
* target availability.

---

## 16.2 Feature Tests

Validated:

* categorical encoding,
* BMI categorization,
* interaction generation,
* smoker/obese flag,
* final feature ordering.

---

## 16.3 Model Tests

Validated:

* baseline model,
* candidate models,
* final model,
* validation metrics,
* held-out test metrics,
* business metric.

---

## 16.4 API Tests

Validated:

* service availability,
* health endpoint,
* valid request payload,
* prediction generation,
* different applicant profiles,
* response structure.

---

## 16.5 Regression Testing

The serving pipeline reuses feature-engineering logic from the training pipeline.

This is specifically intended to reduce the risk of:

> **Training-serving skew**

where features generated during production inference differ from the features used during model training.

---

# 17. SDLC Phase 12 — Documentation & Handover

The project maintains separate operational and technical documentation.

## README

Designed for:

* installation,
* execution,
* API startup,
* repository navigation,
* output discovery.

## PROJECT_DOCUMENTATION

Designed for:

* architecture,
* data dictionary,
* business context,
* modeling decisions,
* evaluation,
* API contract,
* monitoring,
* maintenance,
* limitations.

## Model Card

Captures:

* intended use,
* limitations,
* evaluation,
* primary features,
* known risks.

## Experiment Log

Records:

* baseline model,
* candidate models,
* validation metrics,
* model-selection rationale.

---

# 18. Model Explainability

The final model's top five features were:

| Rank | Feature                  | Importance |
| ---: | ------------------------ | ---------: |
|    1 | `smoker`                 |     0.4156 |
|    2 | `smoker_bmi_interaction` |     0.3520 |
|    3 | `age`                    |     0.0926 |
|    4 | `children`               |     0.0308 |
|    5 | `bmi`                    |     0.0140 |

The first two features account for:

**76.76% of total feature importance.**

This is consistent with the EDA finding that smoking and its interaction with BMI dominate the observed cost signal.

---

# 19. Feature Engineering Validation

An important model-development observation is that the binary feature:

```text
is_smoker_and_obese
```

had importance:

**0.0076**

while:

```text
smoker_bmi_interaction
```

had importance:

**0.3520**

This indicates that the model obtained more predictive information from the continuous interaction than from the binary summary flag.

This is a useful engineering finding:

> A feature hypothesis can be correct even when one particular engineered representation of that hypothesis contributes relatively little to the final model.

---

# 20. SDLC Phase 13 — Monitoring

## 20.1 Feature Drift

PSI was calculated for:

* age,
* BMI,
* smoker,
* smoker × BMI interaction.

Observed PSI:

**0.002–0.056**

The monitored distributions were classified as stable for this dataset-based simulation.

---

## 20.2 Performance Monitoring

The monitoring process compares current:

**% predictions within ±20%**

against the stored baseline.

Retraining recommendation threshold:

**performance decrease > 0.05**

Current monitoring execution:

**No retraining triggered.**

---

## 20.3 Critical Monitoring Limitation

The source dataset contains no timestamp.

Therefore, the monitoring implementation cannot establish:

* year-over-year medical inflation,
* temporal claims drift,
* changing healthcare utilization,
* changing smoker prevalence,
* temporal population shift.

The current monitoring result confirms that the **monitoring mechanism executes**, not that a production model is proven stable over time.

---

# 21. SDLC Phase 14 — Maintenance & Continuous Improvement

## 21.1 Retraining Conditions

A production system should consider retraining when:

* predictive performance deteriorates,
* feature distributions shift materially,
* new claims data becomes available,
* cost patterns change,
* population composition changes,
* business requirements change.

---

## 21.2 Data Expansion

The production model should eventually incorporate richer information such as:

* prior claims,
* medical history,
* healthcare utilization,
* diagnosis information,
* prescription history,
* geographic cost variation,
* historical policy information,
* temporal features.

---

## 21.3 Model Improvement

Future experiments may include:

* cross-validation,
* hyperparameter optimization,
* alternative boosting algorithms,
* ensemble models,
* calibration,
* segment-specific error analysis,
* uncertainty estimation.

---

# 22. Model Risk & Governance Considerations

A production insurance model would require considerably more governance than demonstrated by this proof-of-concept.

Important governance considerations include:

### Data Representativeness

The dataset contains only 1,338 individuals and may not represent a production insurer's policyholder population.

### Temporal Validity

The dataset is a single historical snapshot.

### Feature Sufficiency

Only six raw predictors are available.

### Segment Performance

Model performance should be evaluated separately across:

* smoker status,
* BMI categories,
* age groups,
* sex,
* region.

### Fairness / Regulatory Review

Any production use of demographic variables would require appropriate legal, regulatory, actuarial, and model-risk review for the relevant jurisdiction.

### Explainability

A production underwriting application may require explanations suitable for business and compliance stakeholders rather than relying solely on tree-based feature importance.

---

# 23. Known Limitations

## 23.1 Dataset Size

Only **1,337 records** remain after deduplication.

This is small compared with a real insurer's policyholder and claims population.

---

## 23.2 Dataset Scope

The dataset is:

* US-only,
* historical,
* single-snapshot,
* limited to six raw predictors.

---

## 23.3 No Medical History

The dataset does not contain:

* diagnoses,
* prior claims,
* procedures,
* medications,
* healthcare utilization.

Therefore, the model should not be interpreted as a complete medical-cost underwriting model.

---

## 23.4 No Temporal Data

No timestamp is available.

Consequently, genuine temporal drift and medical-cost inflation cannot be evaluated.

---

## 23.5 Segment Concentration

A substantial portion of the predictive signal is concentrated around smoking and the smoking × BMI interaction.

The non-smoker majority therefore requires additional segment-level validation before any production interpretation.

---

## 23.6 Business Impact Generalization

The measured:

**74.27% reserving-error reduction**

is an observed result for the held-out dataset.

It must **not** be presented as a guaranteed production financial impact.

---

# 24. Reproducibility & Artifact Management

The project produces the following artifacts:

```text
outputs/
├── engineered_data.csv
├── eda_summary.png
├── experiment_log.csv
├── business_validation.json
├── feature_importance.csv
├── cost_model.joblib
├── feature_columns.joblib
└── model_card.json
```

These artifacts allow a reviewer to inspect:

* transformed data,
* exploratory findings,
* experiment comparison,
* business validation,
* feature importance,
* trained model,
* inference schema,
* model limitations.

---

# 25. Productionization Gap Analysis

The current implementation demonstrates the complete ML lifecycle but is intentionally distinguishable from a production insurance platform.

| Area              | Current POC           | Production Requirement                    |
| ----------------- | --------------------- | ----------------------------------------- |
| Dataset           | 1,338 records         | Large representative claims population    |
| Data pipeline     | Local scripts         | Automated governed pipeline               |
| Model registry    | File artifact         | Versioned model registry                  |
| Deployment        | Local FastAPI         | Containerized cloud service               |
| Authentication    | Not implemented       | Enterprise IAM                            |
| Monitoring        | PSI + performance     | Full observability                        |
| Drift             | Snapshot simulation   | Continuous temporal monitoring            |
| Retraining        | Trigger simulation    | Automated governed retraining             |
| Explainability    | Feature importance    | Explainability framework                  |
| Governance        | Model card            | Formal model-risk governance              |
| Testing           | Script/API validation | CI/CD automated testing                   |
| Data quality      | Script-level          | Production DQ framework                   |
| Security          | POC                   | Enterprise security controls              |
| Auditability      | Local artifacts       | Centralized audit trail                   |
| Regulatory review | Not performed         | Required before applicable production use |

---

# 26. Operational Runbook

## Training

```bash
python data_loader.py
python eda.py
python clean_and_engineer.py
python train_model.py
```

## Monitoring

```bash
python monitor.py
```

## API

```bash
uvicorn app:app --reload
```

## Health Check

```text
GET /health
```

## Prediction

```text
POST /estimate
```

---

# 27. Acceptance Criteria

The project satisfies the following development acceptance criteria:

| Criterion                          | Status |
| ---------------------------------- | ------ |
| Real dataset used                  | ✓      |
| Duplicate handling documented      | ✓      |
| EDA completed                      | ✓      |
| Interaction discovered from EDA    | ✓      |
| Feature engineering implemented    | ✓      |
| Baseline established               | ✓      |
| Multiple models evaluated          | ✓      |
| Model selection metric defined     | ✓      |
| Held-out test evaluation completed | ✓      |
| Business validation completed      | ✓      |
| API deployed                       | ✓      |
| API tested with different inputs   | ✓      |
| Monitoring implemented             | ✓      |
| Retraining trigger implemented     | ✓      |
| Limitations documented             | ✓      |
| Model artifact generated           | ✓      |
| Technical handover completed       | ✓      |

---

# 28. Final SDLC Traceability

| Phase                              | Objective                                     | Primary Evidence                        |
| ---------------------------------- | --------------------------------------------- | --------------------------------------- |
| **1. Business Problem Definition** | Define insurance cost estimation problem      | Business objective                      |
| **2. Requirements Analysis**       | Define functional/non-functional requirements | Requirements specification              |
| **3. Feasibility Analysis**        | Establish technical/business feasibility      | Regression + API architecture           |
| **4. Project Planning**            | Define lifecycle and deliverables             | Project pipeline                        |
| **5. Data Acquisition & Quality**  | Acquire and validate data                     | 1,338 → 1,337                           |
| **6. EDA**                         | Identify cost drivers                         | Smoker/BMI interaction                  |
| **7. Preparation & Engineering**   | Create modeling dataset                       | 15 features                             |
| **8. Model Development**           | Build baseline/candidates                     | LR/RF/XGBoost                           |
| **9. Evaluation & Selection**      | Select model using business metric            | XGBoost                                 |
| **10. Deployment**                 | Expose model                                  | FastAPI                                 |
| **11. Testing**                    | Validate system                               | Data/model/API tests                    |
| **12. Documentation**              | Enable handover                               | README/model card/project documentation |
| **13. Monitoring**                 | Detect drift/performance deterioration        | PSI + threshold                         |
| **14. Maintenance**                | Define lifecycle after deployment             | Retraining strategy                     |

---

# 29. Executive Conclusion

This project demonstrates an end-to-end machine-learning SDLC for **insurance cost / claim-severity prediction**, progressing from business problem definition through data acquisition, exploratory analysis, feature engineering, model development, validation, API deployment, monitoring, and maintenance planning.

The most significant analytical finding was the **smoker × BMI interaction**. Non-smoking obese individuals averaged **$8,853** in annual charges compared with **$41,693** for smoking obese individuals. This empirical observation directly informed the interaction feature used by the final model.

The final XGBoost model achieved on the held-out test population:

* **MAE:** $2,338.17
* **MAPE:** 19.35%
* **R²:** 0.8584
* **Within ±20%:** 69.65%

On the same 201-person test cohort, the individualized model produced **$469,971.31** of aggregate absolute error compared with **$1,826,446.39** from the flat training-average baseline, corresponding to a measured difference of **$1,356,475.07 (74.27%)**.

This business-validation result is directly calculated from the observed dollar-valued `charges` in the dataset. It is therefore a **measured result for this test cohort**, rather than an externally assumed financial-impact estimate.

At the same time, the project explicitly recognizes that the dataset is a **small, US-only, single-snapshot dataset without medical-history or temporal claims information**. Consequently, the model should be treated as a **production-architecture demonstration and proof of concept**, not as an actuarially validated production underwriting model.


