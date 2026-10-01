# AeroMaintain AI — System Architecture

## 1. Purpose

AeroMaintain AI is an end-to-end aircraft maintenance intelligence portfolio system built to demonstrate how predictive machine learning, maintenance history, semantic retrieval, and agent orchestration can be combined into a traceable decision-support workflow.

The system is designed around the following investigation:

> Given the latest available telemetry for an aircraft component, estimate its near-term risk, recover relevant maintenance history, retrieve supporting engineering evidence, produce a traceable recommendation, and propose a human-gated workflow action.

AeroMaintain is not designed to autonomously determine airworthiness, prescribe maintenance actions, or replace qualified maintenance personnel.

All aircraft, component, telemetry, maintenance, failure, and knowledge-base information used by the project is synthetic.

---

# 2. Architecture Goals

The architecture was designed around several engineering goals.

## 2.1 Traceability

The final investigation should not be a black-box generated answer.

A user should be able to inspect:

- the component being investigated,
- the model risk score,
- the decision threshold,
- the maintenance events returned,
- the retrieval query,
- the engineering passages retrieved,
- the deterministic recommendation,
- the proposed workflow action,
- the human-approval and execution state,
- the limitations attached to the investigation.

Each layer therefore exposes structured intermediate outputs.

## 2.2 Leakage-Aware Machine Learning

Predictive features must represent information that could reasonably be available at inference time.

Simulation-only variables and future information are excluded from production model features.

## 2.3 Temporal Evaluation

Aircraft component observations are time-dependent.

The ML evaluation therefore uses temporal partitions and purge gaps rather than randomly mixing observations from the same synthetic year across train and test sets.

## 2.4 Modular AI Components

Prediction, retrieval, maintenance-history access, recommendation logic, action proposal logic, and orchestration are kept as explicit system responsibilities.

This allows each subsystem to be tested and evaluated independently.

## 2.5 Local-First Execution

Core inference and retrieval do not depend on a paid external LLM or embedding API.

The deployed workflow can operate with:

- a persisted scikit-learn model,
- local Parquet data,
- local synthetic documentation,
- TF-IDF / LSA retrieval,
- deterministic Python orchestration.

## 2.6 Human-Gated Decision Support

AeroMaintain extends investigation support with a deterministic recommendation and a proposed workflow action.

The system intentionally separates **recommendation** from **execution**. A proposed action remains human-gated and is not treated as a completed maintenance action.

The current implementation does not modify an external maintenance system, determine airworthiness, or authorize maintenance execution.

---

# 3. High-Level Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    SYNTHETIC FLEET                          │
│                                                             │
│  Aircraft ─ Components ─ Telemetry ─ Maintenance Events    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 DATABRICKS LAKEHOUSE                        │
│                                                             │
│   Bronze  ─────►  Silver  ─────►  Gold Predictive Features │
│    Raw           Quality           Leakage-Safe Features    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  PREDICTIVE ML                              │
│                                                             │
│   Preprocessing ─► Logistic Regression ─► Risk Score       │
│                                                             │
│                  MLflow Experiment Tracking                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │
              ┌────────────┴─────────────┐
              │                          │
              ▼                          ▼
┌──────────────────────────┐   ┌──────────────────────────────┐
│   MAINTENANCE HISTORY    │   │   MAINTENANCE KNOWLEDGE     │
│                          │   │          BASE                │
│ Local maintenance events │   │                              │
└─────────────┬────────────┘   │ Synthetic Markdown Docs      │
              │                │           │                  │
              │                │           ▼                  │
              │                │    Chunking / Indexing       │
              │                │           │                  │
              │                │           ▼                  │
              │                │      LSA Retrieval           │
              │                └───────────┬──────────────────┘
              │                            │
              └────────────┬───────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 MAINTENANCE AGENT                           │
│                                                             │
│ Observe ─ Risk ─ History ─ Evidence ─ Recommend            │
│                         │                                   │
│                         ▼                                   │
│              Propose Workflow Action                        │
│                         │                                   │
│                         ▼                                   │
│                 Human Approval Gate                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
               ┌───────────┴───────────┐
               │                       │
               ▼                       ▼
┌──────────────────────────┐   ┌──────────────────────────────┐
│        FastAPI           │   │         Streamlit            │
│                          │   │                              │
│ Programmatic Interface   │   │ Analyst-Facing Interface     │
└──────────────────────────┘   └──────────────────────────────┘
```

---

# 4. Repository Architecture

The implementation follows the same separation of concerns as the conceptual architecture.

```text
src/aeromaintain/
│
├── synthetic/
│   ├── generator.py
│   └── maintenance_docs.py
│
├── features/
│   └── build_features.py
│
├── models/
│   ├── train.py
│   └── predict.py
│
├── rag/
│   ├── ingest.py
│   └── retrieve.py
│
├── agents/
│   ├── tools.py
│   └── maintenance_agent.py
│
├── api/
│   └── main.py
│
└── config.py
```

Supporting platform code lives outside the core package:

```text
databricks/
    01_bronze.py
    02_silver.py
    03_gold.py
    04_ml_training.py

evaluation/
    evaluate_retrieval.py
    retrieval_queries.json

app/
    streamlit_app.py
```

This separation prevents the user interface, data engineering pipeline, retrieval evaluation, and inference implementation from collapsing into a single application script.

---

# 5. Synthetic Data Architecture

## 5.1 Fleet

The current synthetic environment contains:

```text
100 aircraft
500 components
5 component categories
```

Each aircraft contains one component from each modeled component category.

The simulation includes aircraft-level utilization variation and hidden component susceptibility.

These variables help create heterogeneous degradation behavior across the fleet.

## 5.2 Component Categories

The knowledge and telemetry layers cover:

```text
HYDRAULIC_PUMP
GENERATOR
AIR_CYCLE_MACHINE
FUEL_PUMP
ACTUATOR
```

## 5.3 Telemetry

The generated telemetry contains component observations such as:

```text
timestamp
aircraft_id
component_id
component_type
flight_cycles
days_since_maintenance
previous_failures
health_index
temperature_c
vibration_mm_s
failure_event
```

The synthetic generator intentionally introduces limited data-quality problems, including:

- missing temperature measurements,
- vibration outliers,
- duplicate observations.

This allows the lakehouse pipeline to demonstrate explicit data-quality handling.

## 5.4 Latent Simulation Variables

Some variables exist to drive synthetic degradation but must not become production model inputs.

Examples include:

```text
health_index
component susceptibility
```

These variables represent information available to the simulator rather than information assumed to be available to an inference system.

They are therefore excluded from the predictive model.

---

# 6. Lakehouse Architecture

The data-engineering layer is implemented in Databricks using a Bronze / Silver / Gold design.

---

## 6.1 Bronze Layer

The Bronze layer preserves raw synthetic telemetry.

Primary purpose:

```text
raw ingestion
+
traceability
```

The layer avoids aggressive transformations so downstream processing can be audited against the source simulation output.

---

## 6.2 Silver Layer

The Silver layer performs data-quality processing.

Operations include:

```text
duplicate removal
temperature missing-value flagging
vibration outlier flagging
failure-event preservation
```

The original simulation produced approximately:

```text
182,865 telemetry rows
```

After duplicate removal:

```text
182,500 telemetry rows
```

The Silver pipeline preserves the observed failure events rather than silently replacing or smoothing them.

Missing and anomalous measurements are represented explicitly through quality indicators.

---

## 6.3 Gold Layer

The Gold layer creates the supervised predictive dataset.

Its responsibilities include:

- joining aircraft metadata,
- joining component metadata,
- calculating component-life features,
- calculating rolling telemetry features,
- calculating maintenance-recency features,
- constructing the future prediction target,
- removing observations without a complete future horizon.

The final Databricks table is:

```text
workspace.aeromaintain.gold_predictive_features
```

---

# 7. Prediction Target

The supervised task asks whether a component experiences a failure event within the next:

```text
30 days
```

Conceptually:

```text
observation at time t
        │
        ▼
inspect future interval
(t, t + 30 days]
        │
        ▼
failure present?
   │          │
  yes         no
   │          │
   1          0
```

The current row's failure event is not used as the prediction target.

The target represents future behavior.

---

# 8. Right Censoring

Observations near the end of the synthetic dataset do not have a complete 30-day future window.

Training on those observations as negative examples would incorrectly imply that no future failure occurred when the future is actually unobserved.

AeroMaintain therefore removes right-censored observations from supervised training.

After censoring:

```text
167,500 eligible observations
4,770 positive observations
~2.85% positive rate
```

This distinction is important for reliability-style prediction problems where the available observation horizon ends before every component's future outcome is known.

---

# 9. Predictive Feature Architecture

The production model uses 20 features.

## 9.1 Categorical Features

```text
aircraft_type
component_type
```

## 9.2 Numeric Features

```text
age_years
utilization_factor
flight_cycles
expected_life_cycles
cycle_life_ratio
days_since_maintenance
previous_failures
temperature_c
vibration_mm_s
temperature_missing_flag
temperature_mean_7d
temperature_mean_30d
temperature_std_7d
temperature_delta_7d
vibration_mean_7d
vibration_mean_30d
vibration_std_7d
vibration_delta_7d
```

---

# 10. Leakage Controls

Leakage prevention is treated as an architectural requirement rather than only a modeling detail.

Variables excluded from production ML include:

```text
health_index
failure_event
future failure counts
hidden susceptibility
simulation-only metadata
```

The vibration outlier flag is retained for data-quality auditing but excluded from the model because its current calculation uses information from the full synthetic year.

This avoids introducing a feature whose definition indirectly depends on future observations.

---

# 11. Temporal Training Architecture

A random split would allow observations from similar components and nearby dates to appear across both training and evaluation sets.

Instead, AeroMaintain uses temporal partitions.

```text
TRAIN
through 2025-08-01
        │
        ▼
30-day purge gap
        │
        ▼
VALIDATION
2025-09-01 → 2025-09-30
        │
        ▼
30-day purge gap
        │
        ▼
TEST
2025-10-31 → 2025-12-01
```

Dataset sizes:

```text
Train       106,500
Validation   15,000
Test         16,000
```

The purge intervals reduce contamination caused by overlapping 30-day prediction horizons.

---

# 12. ML Pipeline

The predictive pipeline is implemented with scikit-learn.

Conceptually:

```text
                    ┌─────────────────────┐
Numeric Features ──►│ Median Imputation   │
                    │ Standard Scaling    │
                    └──────────┬──────────┘
                               │
                               ├──────────────┐
                               │              │
                    ┌──────────▼──────────┐   │
Categorical ───────►│ Most-Frequent      │   │
Features            │ Imputation         │   │
                    │ One-Hot Encoding   │   │
                    └──────────┬──────────┘   │
                               │              │
                               └──────┬───────┘
                                      ▼
                             Logistic Regression
                                      │
                                      ▼
                                  Risk Score
```

The preprocessing pipeline is fitted only on training data.

---

# 13. Model Selection

Two initial predictive baselines were evaluated:

```text
Balanced Logistic Regression
Random Forest
```

Validation results:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| PR-AUC | 0.0496 | 0.0353 |
| Best F1 | 0.1168 | 0.0889 |

The class-balanced Logistic Regression model was selected.

The selected validation threshold is approximately:

```text
0.7248
```

The threshold was selected from validation data rather than optimized against the held-out test set.

---

# 14. Held-Out Test Evaluation

The selected model achieved:

```text
PR-AUC       0.0713
ROC-AUC      0.6811
Precision    0.0802
Recall       0.4006
F1           0.1337
```

Test prevalence / random PR baseline:

```text
0.0393
```

Observed PR-AUC lift over the random baseline:

```text
~1.81x
```

Confusion matrix:

```text
                 Predicted
               Negative Positive

Actual Negative  12,481   2,890
Actual Positive     377     252
```

The model is not described as production-ready.

Its role in the architecture is to provide a measurable synthetic risk-ranking signal to the downstream investigation workflow.

---

# 15. Model Persistence and MLflow

Training experiments are tracked through MLflow in Databricks.

The selected model is persisted into a local inference bundle.

The bundle includes:

- fitted preprocessing pipeline,
- fitted classifier,
- feature definitions,
- decision threshold,
- prediction horizon,
- model metadata.

The persisted model was reloaded and compared against the original fitted model.

A 100-row inference comparison produced:

```text
maximum prediction difference = 0.0
```

This validates serialization consistency for the tested sample.

---

# 16. Local Inference Architecture

Cloud training infrastructure is not required for normal application inference.

The application loads:

```text
artifacts/predictive_maintenance_bundle.joblib
```

and reconstructs the latest component features from the synthetic runtime data.

Inference returns structured information including:

```text
risk_score
threshold
above_threshold
prediction_horizon_days
model_type
interpretation
calibrated_probability
data_type
```

The current Logistic Regression output is treated as:

```text
uncalibrated risk score
```

not:

```text
probability of failure
```

---

# 17. Knowledge Architecture

The project includes a synthetic maintenance knowledge base.

Documents exist for:

```text
HYDRAULIC_PUMP
GENERATOR
AIR_CYCLE_MACHINE
FUEL_PUMP
ACTUATOR
```

The documents describe synthetic:

- component purpose,
- condition indicators,
- investigation considerations,
- maintenance context.

They are not OEM manuals and do not contain approved maintenance instructions.

---

# 18. Document Ingestion

The ingestion pipeline reads the Markdown knowledge documents and converts them into retrieval chunks.

Conceptually:

```text
Markdown Documents
        │
        ▼
Document Parsing
        │
        ▼
Section / Chunk Extraction
        │
        ▼
Structured Chunk Collection
        │
        ▼
Retrieval Index
```

The current knowledge base produces:

```text
30 chunks
```

---

# 19. Retrieval Architecture

Two retrieval strategies were implemented.

## 19.1 TF-IDF Baseline

The first implementation provides lexical retrieval using TF-IDF.

It establishes a simple measurable baseline.

## 19.2 LSA Semantic Retrieval

The second implementation applies:

```text
TF-IDF
  │
  ▼
Truncated SVD
  │
  ▼
Latent Semantic Representation
  │
  ▼
Similarity Ranking
```

This approach provides fully local latent semantic retrieval without requiring an external embedding API.

It should be described accurately as:

```text
LSA semantic retrieval
```

rather than transformer-based embeddings.

---

# 20. Retrieval Evaluation

A fixed synthetic benchmark contains 10 retrieval queries.

Measured results:

| Retriever | Hit@1 | Hit@3 | MRR |
|---|---:|---:|---:|
| TF-IDF | 0.500 | 0.800 | 0.633 |
| LSA | 0.700 | 1.000 | 0.850 |

Because LSA improved all three benchmark metrics, it became the default retrieval strategy used by the maintenance agent.

The benchmark is intentionally described as a synthetic project benchmark rather than evidence of performance on real aviation documentation.

---

# 21. Maintenance Tools Layer

The agent interacts with system capabilities through a tools abstraction.

The tools layer provides access to:

```text
predictive risk inference
maintenance history
component metadata
semantic retrieval
```

This separates orchestration logic from individual implementations.

Conceptually:

```text
Maintenance Agent
       │
       ▼
MaintenanceTools
       │
       ├──► Predictor
       │
       ├──► Maintenance Events
       │
       ├──► Component Metadata
       │
       └──► LSA Retriever
```

The predictor is loaded lazily so retrieval-oriented operations do not unnecessarily initialize model inference.

---

# 22. Maintenance Agent Architecture

The maintenance agent is deterministic, evidence-driven, and human-gated.

It does not require an LLM to decide the workflow. It combines explicit tool outputs with deterministic decision rules so that the recommendation and proposed action can be inspected independently.

The orchestration sequence is:

```text
1. Validate component
          │
          ▼
2. Calculate predictive risk
          │
          ▼
3. Retrieve maintenance history
          │
          ▼
4. Resolve component type
          │
          ▼
5. Build investigation query
          │
          ▼
6. Retrieve component-specific evidence
          │
          ▼
7. Build deterministic recommendation
          │
          ▼
8. Propose workflow action
          │
          ▼
9. Expose human-approval state
          │
          ▼
10. Assemble summary + limitations
```

At the decision layer, the current deterministic policy distinguishes two workflow recommendations:

```text
risk below configured threshold
        │
        ▼
CONDITION_MONITORING
priority = ROUTINE
        │
        ▼
CREATE_MONITORING_CASE
```

and:

```text
risk at / above configured threshold
        │
        ▼
ENGINEERING_REVIEW
priority = ELEVATED
        │
        ▼
CREATE_ENGINEERING_REVIEW_CASE
```

These are **workflow recommendations and proposals**, not approved aircraft maintenance instructions.

Every proposed action is explicitly represented as:

```text
status = PROPOSED
approval_status = HUMAN_APPROVAL_REQUIRED
execution_status = NOT_EXECUTED
external_system_modified = False
```

The resulting investigation contract contains:

```text
component_id
component_type
risk
maintenance_history
retrieval_query
evidence
recommendation
action_proposal
summary
limitations
```

The limitations contract also exposes the safety boundary explicitly, including that autonomous maintenance decisions and real-world action execution are disabled.

This contract is shared by the downstream service and interface layers.

---

# 23. Why Deterministic Orchestration?

A general-purpose language model was not required to implement the core investigation workflow.

This was intentional.

The workflow has a known structure:

```text
risk
+
history
+
retrieved evidence
=
traceable decision context
        │
        ▼
deterministic recommendation
        │
        ▼
proposed workflow action
        │
        ▼
human approval required
```

Using explicit orchestration provides:

- predictable execution,
- easier testing,
- transparent data flow,
- no external LLM dependency,
- no paid inference requirement,
- easier debugging,
- clearer safety boundaries.

An LLM could later be introduced as an optional summarization layer without replacing the structured evidence contract.

---

# 24. API Architecture

FastAPI exposes the core maintenance intelligence workflow.

Endpoints include:

```text
GET /
GET /health
GET /investigate/{component_id}
GET /components/{component_id}/risk
```

Conceptually:

```text
Client
  │
  ▼
FastAPI
  │
  ▼
MaintenanceAgent / Predictor
  │
  ▼
Recommendation + Action Proposal
  │
  ▼
Structured JSON Response
```

The API allows the AI workflow to be consumed independently from Streamlit.

---

# 25. Streamlit Architecture

The Streamlit dashboard is the human-facing investigation interface.

It imports the core maintenance agent directly.

Current application path:

```text
Streamlit
    │
    ▼
MaintenanceAgent
    │
    ├── Predictor
    ├── Maintenance History
    ├── LSA Retrieval
    ├── Recommendation Policy
    └── Human-Gated Action Proposal
```

The current Streamlit application does **not** make an HTTP request to the local FastAPI service.

FastAPI exists as a separate service interface demonstrating how the same core system can be exposed programmatically.

This distinction prevents the architecture documentation from implying a network dependency that does not exist in the current application.

---

# 26. Streamlit Investigation Experience

The interface exposes five primary intelligence layers:

```text
Predictive Risk
Maintenance History
Evidence Retrieval
Agent Recommendation
Human-Gated Action Proposal
```

The result view provides:

- component identity,
- component category,
- risk score,
- selected threshold,
- threshold status,
- prediction horizon,
- maintenance-event history,
- retrieved evidence,
- recommendation code and priority,
- proposed workflow action,
- approval and execution state,
- investigation summary,
- explicit limitations.

The UI displays scores numerically, for example:

```text
0.7145
```

rather than presenting:

```text
71.45% probability of failure
```

because the current score is not probability calibrated.

---

# 27. Testing Architecture

Tests cover the major system layers.

The test suite includes coverage for:

```text
synthetic generation
feature engineering
retrieval
agent tools
maintenance agent
FastAPI
```

Current checkpoint:

```text
60 tests passed
```

The suite now explicitly tests the feature-engineering contract, agent recommendation/action-proposal behavior, API exposure of the agentic contract, and human-gated safety boundaries.

Tests are intended to verify both isolated subsystem behavior and cross-layer contracts.

---

# 28. Deployment Runtime

The local/deployment runtime contains the small assets needed for inference:

```text
artifacts/
└── predictive_maintenance_bundle.joblib

data/raw/
├── aircraft.parquet
├── components.parquet
├── maintenance.parquet
└── telemetry.parquet
```

The telemetry dataset is approximately:

```text
1.4 MB
```

The model artifact is only a few kilobytes.

Because these assets are synthetic and small, they can be included with the portfolio repository so the demonstration does not depend on a live Databricks connection.

Databricks remains the training and lakehouse environment, while application inference can execute independently.

---

# 29. Training vs. Serving Separation

A key architectural boundary is the separation between training infrastructure and serving infrastructure.

## Training Path

```text
Synthetic Generator
       │
       ▼
Databricks Bronze
       │
       ▼
Databricks Silver
       │
       ▼
Databricks Gold
       │
       ▼
ML Training + MLflow
       │
       ▼
Persisted Model Bundle
```

## Serving Path

```text
Persisted Model Bundle
       +
Synthetic Runtime Data
       +
Synthetic Knowledge Base
       │
       ▼
Local Feature Builder
       │
       ▼
Predictor + Retriever
       │
       ▼
Maintenance Agent
       │
       ▼
Recommendation + Proposed Action
       │
       ▼
Human Approval Boundary
       │
       ├──► FastAPI
       │
       └──► Streamlit
```

This prevents the demonstration application from requiring the full training platform to remain online.

---

# 30. Failure and Safety Boundaries

AeroMaintain operates in a safety-sensitive domain, so the architecture deliberately avoids several claims.

The system does not provide:

```text
airworthiness determination
approved maintenance procedures
minimum equipment decisions
return-to-service authorization
regulatory compliance decisions
OEM troubleshooting instructions
autonomous maintenance actions
```

The system instead provides:

```text
synthetic predictive signal
+
synthetic event history
+
synthetic retrieved evidence
+
deterministic recommendation
+
human-gated workflow proposal
+
traceable investigation context
```

for portfolio demonstration.

---

# 31. Current Architecture Limitations

The current system has several important limitations.

## Synthetic Data

All operational behavior is simulated.

Performance measurements therefore describe the synthetic environment only.

## Simplified Failure Process

The generator models degradation using intentionally simplified assumptions rather than real component physics.

## Small Retrieval Benchmark

Retrieval evaluation currently contains only 10 synthetic queries.

## Uncalibrated Risk

The selected Logistic Regression score is used for ranking and thresholding but is not presented as a calibrated failure probability.

## Local Runtime Data

The demonstration loads local Parquet files rather than querying a production feature store.

## No Authentication

The portfolio API and Streamlit interface do not currently implement production authentication or authorization.

## No Production Monitoring

The project does not yet include deployed drift monitoring, service-level telemetry, or alerting.

## No Real Maintenance Documentation

No OEM or regulatory manuals are included.

## No Real-World Action Execution

The recommendation layer can propose a monitoring or engineering-review workflow case, but the current system does not create a case in an external maintenance platform, modify aircraft records, execute maintenance, or grant return-to-service authority.

Every action proposal remains `PROPOSED`, requires human approval, and is reported as `NOT_EXECUTED`.

---

# 32. Production Evolution

A real-world system inspired by this architecture would require additional layers.

Conceptually:

```text
Operational Aircraft Data
          │
          ▼
Governed Streaming / Batch Ingestion
          │
          ▼
Validated Feature Platform
          │
          ▼
Versioned + Calibrated Risk Models
          │
          ├──────────────┐
          ▼              ▼
Approved Technical   Maintenance
Documentation        Records
          │              │
          └──────┬───────┘
                 ▼
      Controlled Retrieval Layer
                 │
                 ▼
       Human-in-the-Loop Workflow
                 │
                 ▼
      Audit / Monitoring / Governance
```

Such a system would require, among other controls:

- validated source systems,
- access control,
- data lineage,
- model governance,
- domain-expert validation,
- calibrated risk communication,
- approved documentation sources,
- audit logging,
- model and data monitoring,
- applicable aviation regulatory compliance.

Those capabilities are outside the scope of the current portfolio implementation.

---

# 33. Architectural Summary

AeroMaintain demonstrates a layered AI engineering architecture:

```text
Synthetic Data
      ↓
Lakehouse Engineering
      ↓
Leakage-Safe Features
      ↓
Temporal Predictive ML
      ↓
Persisted Local Inference
      ↓
Semantic Retrieval
      ↓
Evidence-Driven Agent
      ↓
Deterministic Recommendation
      ↓
Human-Gated Action Proposal
      ↓
API + Analyst Interface
```

The central architectural principle is:

> **AI output should remain measurable, inspectable, and traceable back to the evidence used to produce it.**

For that reason, AeroMaintain keeps predictive scoring, historical records, retrieved evidence, recommendation logic, action proposals, approval state, orchestration, and presentation as separate layers rather than hiding the entire workflow behind a single generated response.

The current architecture therefore follows the decision-support path:

```text
Observe
  ↓
Risk Assess
  ↓
Retrieve Evidence
  ↓
Recommend
  ↓
Propose Action
  ↓
Human Approval
```

The final boundary is deliberate: **proposal is not execution**.