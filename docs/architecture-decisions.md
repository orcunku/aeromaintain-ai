# AeroMaintain AI — Architecture Decision Records

## Purpose

This document records the major engineering decisions behind AeroMaintain AI.

The goal is not only to document **what** the system does, but also **why** particular approaches were selected, which alternatives were considered, and what trade-offs remain.

AeroMaintain is a synthetic portfolio system operating in a safety-sensitive domain. Decisions therefore prioritize:

- traceability,
- measurable evaluation,
- leakage prevention,
- reproducibility,
- modularity,
- local-first execution,
- explicit limitations,
- human decision support.

---

# ADR-001 — Use Synthetic Aviation Data

**Status:** Accepted

## Context

Real aircraft telemetry, component reliability records, and maintenance histories are generally proprietary, sensitive, difficult to obtain, and subject to operational and regulatory constraints.

A portfolio project still needs sufficiently structured data to demonstrate:

- fleet-level behavior,
- component degradation,
- maintenance events,
- telemetry trends,
- failures,
- predictive modeling,
- retrieval,
- agent orchestration.

## Decision

AeroMaintain uses a custom synthetic fleet generator.

The current environment models:

```text
100 aircraft
500 components
5 component categories
one synthetic year of telemetry
maintenance events
component degradation
failure behavior
controlled data-quality issues
```

The generator includes hidden susceptibility and aircraft-utilization variation to create heterogeneous component behavior.

## Alternatives Considered

### Public generic predictive-maintenance dataset

Advantages:

- real published benchmark,
- easier comparison with other projects.

Disadvantages:

- less aviation-specific system design,
- reduced control over event history and data generation,
- harder to demonstrate the full end-to-end architecture.

### Proprietary aviation data

Rejected because appropriate data was not available for this portfolio project.

## Consequences

Positive:

- fully reproducible environment,
- no confidential operational data,
- complete control over failure and maintenance simulation,
- ability to test the entire engineering pipeline.

Negative:

- measured model performance does not establish real-world aviation performance,
- synthetic failure mechanisms are simplified,
- real fleet behavior may differ substantially.

## Guardrail

All model and retrieval results must be described as results on **synthetic data**.

---

# ADR-002 — Separate Simulation Variables from Model Features

**Status:** Accepted

## Context

The simulator contains variables that help generate degradation behavior.

Examples include:

```text
health_index
hidden component susceptibility
```

These variables make the synthetic system more realistic from the simulator's perspective, but they would not necessarily exist in an operational inference environment.

Using them in the predictive model would make the ML task artificially easy.

## Decision

Simulation-only latent variables are excluded from production model features.

The model receives only the defined inference-time feature contract.

## Alternatives Considered

### Use all available columns

Rejected because it would allow the model to exploit information available to the simulator rather than the intended prediction system.

## Consequences

The prediction problem becomes more difficult but more representative of an inference architecture.

This also creates a clear separation between:

```text
simulation truth
```

and:

```text
model-observable information
```

---

# ADR-003 — Use Bronze / Silver / Gold Data Layers

**Status:** Accepted

## Context

The project needs to demonstrate more than direct model training from a local CSV or DataFrame.

A production-style ML workflow benefits from explicit separation between:

- raw data,
- cleaned data,
- predictive features.

## Decision

Use a Databricks Bronze / Silver / Gold architecture.

```text
Bronze
Raw ingestion and traceability

Silver
Deduplication and data-quality handling

Gold
Predictive feature engineering and target construction
```

## Alternatives Considered

### Single preprocessing notebook

Advantages:

- faster implementation,
- fewer files.

Disadvantages:

- weaker lineage,
- difficult to inspect intermediate transformations,
- data engineering and modeling become tightly coupled.

## Consequences

The system contains more pipeline code, but each transformation stage has a clearer responsibility.

---

# ADR-004 — Preserve Data-Quality Signals

**Status:** Accepted

## Context

Synthetic telemetry intentionally includes:

- missing temperature values,
- vibration outliers,
- duplicate records.

Simply cleaning all irregularities away would hide useful information about data quality.

## Decision

Remove exact duplicate component/timestamp observations where appropriate while preserving explicit quality indicators such as missing-value flags.

Observed failure events are preserved.

## Consequences

The pipeline distinguishes:

```text
data cleaning
```

from:

```text
data-quality evidence
```

rather than treating every anomaly as something that should silently disappear.

---

# ADR-005 — Predict Future Failure Within a 30-Day Horizon

**Status:** Accepted

## Context

Using the current row's failure state as the target would turn the problem into detection rather than forward-looking predictive maintenance.

The project needs a target that represents future component behavior.

## Decision

Define the supervised target as whether a component experiences a failure event within the next:

```text
30 days
```

## Alternatives Considered

### Same-row failure classification

Rejected because it does not represent a future risk task.

### Very short prediction horizon

Could make prediction easier but provides less forward-looking maintenance context.

### Much longer prediction horizon

Would increase target overlap and censoring while making the relationship between current condition and future outcome less direct.

## Consequences

Feature engineering and dataset splitting must account for the 30-day future window.

---

# ADR-006 — Remove Right-Censored Training Observations

**Status:** Accepted

## Context

Near the end of the dataset, some observations do not have a complete 30-day future window.

Labeling these observations as negative would incorrectly treat an unknown future as a known absence of failure.

## Decision

Exclude observations whose full prediction horizon is not observable.

## Consequences

The supervised dataset becomes smaller but the target semantics remain valid.

Current eligible dataset:

```text
167,500 observations
4,770 positives
~2.85% positive rate
```

---

# ADR-007 — Use Temporal Splits Instead of Random Splits

**Status:** Accepted

## Context

Telemetry observations are time-dependent.

Random splitting could place nearby observations from the same components into both training and evaluation datasets.

That would produce an evaluation that is less representative of future inference.

## Decision

Use chronological train, validation, and test periods.

```text
Train:
through 2025-08-01

Validation:
2025-09-01 → 2025-09-30

Test:
2025-10-31 → 2025-12-01
```

## Alternatives Considered

### Random train/test split

Rejected because temporal mixing can produce optimistic evaluation in this problem.

## Consequences

The model must generalize forward in synthetic time rather than interpolate across randomly mixed observations.

---

# ADR-008 — Add Purge Gaps Between ML Splits

**Status:** Accepted

## Context

The prediction target uses a 30-day future horizon.

Observations near a split boundary can therefore have overlapping target windows even if their feature timestamps are on different sides of the split.

## Decision

Introduce 30-day purge gaps between:

```text
Train → Validation

Validation → Test
```

## Consequences

Some observations are intentionally unused for model fitting or evaluation.

The benefit is cleaner separation between future target windows.

---

# ADR-009 — Exclude Full-Year Vibration Outlier Flag from ML Features

**Status:** Accepted

## Context

The Silver layer includes a vibration outlier flag for data-quality auditing.

Its current threshold is derived using information from the broader synthetic dataset.

Using that flag directly as a model feature could therefore introduce future information into earlier observations.

## Decision

Retain the field for data-quality inspection but exclude it from the predictive model.

## Alternatives Considered

### Keep the flag as a predictive feature

Rejected under the current calculation method.

### Recalculate the flag using training-only or historical windows

Valid future improvement, but not required for the current baseline.

## Consequences

The project demonstrates that a useful data-quality field is not automatically a valid predictive feature.

---

# ADR-010 — Start with Interpretable Predictive Baselines

**Status:** Accepted

## Context

The synthetic dataset is imbalanced, and the project objective is to build a measurable end-to-end AI system rather than maximize a leaderboard metric.

## Decision

Evaluate:

```text
class-balanced Logistic Regression
Random Forest
```

before introducing more complex models.

## Results

Validation PR-AUC:

```text
Logistic Regression    0.0496
Random Forest          0.0353
```

Validation best F1:

```text
Logistic Regression    0.1168
Random Forest          0.0889
```

The Logistic Regression model was selected.

## Alternatives Considered

Potential future models include:

- gradient boosting,
- survival models,
- calibrated ensembles,
- sequence models.

These were not required to validate the current architecture.

## Consequences

The predictive layer remains computationally lightweight and easy to persist locally.

---

# ADR-011 — Select the Decision Threshold on Validation Data

**Status:** Accepted

## Context

A default threshold of `0.5` is not necessarily appropriate for an imbalanced risk-ranking problem.

Selecting a threshold using test data would contaminate the held-out evaluation.

## Decision

Select the operational demonstration threshold using the validation set.

Current selected threshold:

```text
~0.7248
```

The test set is used only after model and threshold selection.

## Consequences

Threshold selection remains separate from final evaluation.

---

# ADR-012 — Report PR-AUC Alongside ROC-AUC

**Status:** Accepted

## Context

The positive class is rare.

In imbalanced classification, ROC-AUC alone can obscure performance on the minority class.

## Decision

Report metrics including:

```text
PR-AUC
ROC-AUC
precision
recall
F1
confusion matrix
positive prevalence
```

The PR-AUC is also compared with the random baseline implied by positive prevalence.

## Consequences

Model evaluation provides more context than a single headline accuracy or ROC-AUC number.

---

# ADR-013 — Treat Model Output as a Risk Score, Not a Calibrated Probability

**Status:** Accepted

## Context

The selected model uses class balancing.

Its `predict_proba` output has not been validated through a probability-calibration procedure.

Displaying a score of:

```text
0.7145
```

as:

```text
71.45% probability of failure
```

would imply a level of probabilistic interpretation that has not been established.

## Decision

Expose the output as an:

```text
uncalibrated risk score
```

and explicitly include:

```text
calibrated_probability = False
```

in inference metadata.

## Alternatives Considered

### Present the raw score as probability

Rejected.

### Add probability calibration

Valid future improvement using an appropriately separated calibration dataset.

## Consequences

UI and API wording must preserve the distinction between ranking score and probability.

---

# ADR-014 — Persist the Complete Inference Pipeline

**Status:** Accepted

## Context

Saving only classifier coefficients is insufficient because inference also depends on:

- imputation,
- scaling,
- categorical encoding,
- feature ordering,
- threshold metadata.

## Decision

Persist an inference bundle containing the fitted preprocessing pipeline, classifier, feature contract, threshold, horizon, and model metadata.

## Consequences

Local inference can reproduce the same transformations used during training.

A reload test confirmed a maximum prediction difference of:

```text
0.0
```

on the tested 100-row sample.

---

# ADR-015 — Use MLflow for Experiment Tracking

**Status:** Accepted

## Context

Model development should retain experiment metadata rather than rely only on notebook output.

## Decision

Use MLflow in Databricks for predictive model experiment tracking.

Tracked information includes:

- model family,
- validation metrics,
- test metrics,
- threshold,
- project metadata,
- intended-use metadata.

## Consequences

Training experiments and model artifacts have a structured tracking layer while the final application remains capable of local inference.

---

# ADR-016 — Separate Training Infrastructure from Serving Infrastructure

**Status:** Accepted

## Context

Requiring Databricks for every dashboard investigation would make the portfolio demo unnecessarily dependent on external infrastructure.

The fitted model and runtime datasets are small.

## Decision

Use Databricks for:

```text
lakehouse processing
feature engineering
training
experiment tracking
```

but allow application inference from local persisted assets.

Serving path:

```text
local Parquet data
+
persisted model
+
local knowledge base
        │
        ▼
Maintenance Agent
```

## Consequences

The application can run independently after the training artifact has been exported.

This improves portability and supports free portfolio deployment.

---

# ADR-017 — Build a Lexical Retrieval Baseline Before Semantic Retrieval

**Status:** Accepted

## Context

A semantic retrieval system should be evaluated against a simpler baseline rather than introduced without comparison.

## Decision

Implement TF-IDF lexical retrieval first.

Measured synthetic benchmark:

```text
Hit@1    0.500
Hit@3    0.800
MRR      0.633
```

## Consequences

The project has a measurable baseline against which later retrieval improvements can be evaluated.

---

# ADR-018 — Use LSA as the Default Semantic Retriever

**Status:** Accepted

## Context

The project initially considered transformer-based local embeddings.

However, the core system should remain:

- free,
- local,
- reproducible,
- lightweight,
- independent of an external model download at runtime.

TF-IDF + Truncated SVD provides a classical latent semantic representation that can be evaluated locally.

## Decision

Implement LSA retrieval using:

```text
TF-IDF
+
TruncatedSVD
```

Measured synthetic benchmark:

```text
Hit@1    0.700
Hit@3    1.000
MRR      0.850
```

## Alternatives Considered

### Transformer sentence embeddings

Potential advantages:

- richer semantic representations,
- stronger general-purpose semantic similarity.

Reasons not selected for the current baseline:

- additional model dependency,
- larger runtime footprint,
- external model acquisition,
- unnecessary complexity for the current synthetic knowledge base.

### TF-IDF only

Retained as the lexical baseline but not selected as the default retriever because LSA performed better on the fixed project benchmark.

## Consequences

AeroMaintain can accurately describe the retrieval system as:

```text
local LSA semantic retrieval
```

It must not be described as transformer embedding retrieval.

---

# ADR-019 — Evaluate Retrieval Independently

**Status:** Accepted

## Context

A retrieval system can appear convincing in individual demonstrations while still performing inconsistently.

## Decision

Create a fixed synthetic retrieval benchmark with expected component/document relevance.

Evaluate:

```text
Hit@1
Hit@3
MRR
```

## Consequences

Retriever selection is based on measured project results rather than visual inspection of a few examples.

## Limitation

The benchmark currently contains only 10 queries and therefore should not be generalized to real aviation documentation.

---

# ADR-020 — Keep the Knowledge Base Synthetic

**Status:** Accepted

## Context

Real aircraft maintenance manuals can be proprietary, copyrighted, controlled, revision-sensitive, or operationally safety-critical.

## Decision

Create project-owned synthetic maintenance documents for the five modeled component categories.

## Consequences

The repository can demonstrate ingestion and retrieval without distributing real OEM maintenance instructions.

All retrieved evidence must remain clearly identified as synthetic.

---

# ADR-021 — Use Component-Aware Evidence Filtering

**Status:** Accepted

## Context

A general semantic search could retrieve text associated with the wrong component category if lexical or latent similarity is high.

For an evidence-driven maintenance investigation, component identity is already known.

## Decision

Use the resolved component type as a retrieval constraint when assembling evidence for an investigation.

## Consequences

The retrieval stage combines:

```text
semantic similarity
+
known component context
```

instead of treating every document as equally eligible.

---

# ADR-022 — Use Deterministic Agent Orchestration

**Status:** Accepted

## Context

The investigation workflow has a well-defined sequence:

```text
validate
predict
retrieve history
resolve component
retrieve evidence
assemble result
```

A general-purpose LLM is not necessary to decide this sequence.

## Decision

Implement the maintenance agent as deterministic Python orchestration.

## Alternatives Considered

### LLM-driven tool selection

Potential advantages:

- flexible natural-language planning,
- dynamic tool selection,
- conversational reasoning.

Disadvantages for the current project:

- external API or local model dependency,
- less deterministic behavior,
- harder automated testing,
- unnecessary variability,
- additional cost or compute,
- weaker traceability for a fixed workflow.

## Consequences

The agent is:

- predictable,
- testable,
- local,
- free to run,
- evidence-driven.

An LLM can later be added as an optional presentation or summarization layer without replacing the structured investigation contract.

---

# ADR-023 — Keep Agent Evidence Structured

**Status:** Accepted

## Context

If the agent returned only a natural-language paragraph, downstream users could not easily distinguish model output from historical records or retrieved evidence.

## Decision

Return a structured investigation contract containing:

```text
component_id
component_type
risk
maintenance_history
retrieval_query
evidence
summary
limitations
```

## Consequences

The UI, API, tests, and future consumers can inspect individual evidence sources directly.

---

# ADR-024 — Expose the Core Workflow Through FastAPI

**Status:** Accepted

## Context

A portfolio system should demonstrate that its core AI capability can be consumed independently from a dashboard.

## Decision

Provide FastAPI endpoints for:

```text
health
component risk
full component investigation
```

## Consequences

The system has a programmatic interface suitable for integration testing and future external clients.

---

# ADR-025 — Let Streamlit Use the Core Agent Directly

**Status:** Accepted

## Context

There are two possible Streamlit architectures:

```text
Streamlit → HTTP → FastAPI → Agent
```

or:

```text
Streamlit → Agent
```

For the current single-process portfolio deployment, forcing the dashboard through a separate HTTP service would add operational complexity without changing the underlying investigation capability.

## Decision

The current Streamlit application imports and invokes the core maintenance agent directly.

FastAPI remains a separate programmatic interface.

## Alternatives Considered

### Force all UI traffic through FastAPI

Advantages:

- strict client/service separation,
- closer to a distributed production deployment.

Disadvantages for the current demo:

- two services must run,
- additional networking and deployment configuration,
- no functional benefit for a single-process portfolio application.

## Consequences

The documentation must not imply that the current Streamlit application communicates with FastAPI over HTTP.

A future distributed deployment can change this boundary without rewriting the core agent.

---

# ADR-026 — Build a Dedicated Intelligence Dashboard

**Status:** Accepted

## Context

A raw Streamlit prototype would demonstrate functionality but would not clearly communicate the architecture or evidence hierarchy to a portfolio reviewer.

## Decision

Create a custom AeroMaintain intelligence interface that emphasizes:

```text
Predictive Risk
Maintenance History
Evidence Retrieval
Agent Investigation
```

The UI also exposes system status and architecture context.

## Consequences

The dashboard acts as both:

- a usable investigation interface,
- a visual explanation of the underlying AI system.

---

# ADR-027 — Avoid Fake Navigation

**Status:** Accepted

## Context

The dashboard sidebar communicates the intelligence layers of the system.

Making static architecture labels appear clickable when they do not navigate would create misleading UX.

## Decision

Present these items as:

```text
Intelligence Layers
```

rather than fake application navigation.

## Consequences

The UI communicates architecture without implying unsupported interactions.

---

# ADR-028 — Include Runtime Assets in the Portfolio Repository

**Status:** Accepted

## Context

The deployed application requires:

```text
predictive model artifact
aircraft data
component data
maintenance data
telemetry data
```

The relevant assets are synthetic and small.

Approximate largest runtime dataset:

```text
telemetry.parquet ≈ 1.4 MB
```

The model bundle is only a few kilobytes.

## Decision

Track the small synthetic runtime assets in Git so the demonstration can run without connecting to Databricks.

## Alternatives Considered

### Regenerate the full synthetic dataset on application startup

Advantages:

- source-only repository,
- explicit reproducibility.

Disadvantages:

- slower startup,
- additional runtime work,
- potential mismatch between deployed data and persisted trained model if generation configuration changes.

### Download assets from external storage

Rejected for the current portfolio deployment because it adds an unnecessary external dependency.

## Consequences

The demo repository is slightly larger but substantially easier to run and deploy.

---

# ADR-029 — Keep Processed Pipeline Outputs Out of Git

**Status:** Accepted

## Context

Not every intermediate dataset needs to be committed simply because runtime assets are included.

## Decision

Continue ignoring:

```text
data/processed/
```

while tracking only the small runtime data required by the application.

## Consequences

The repository contains enough data to run the demonstration without becoming a storage location for every generated intermediate artifact.

---

# ADR-030 — Use Automated Tests Across System Layers

**Status:** Accepted

## Context

A multi-layer AI system can fail at contracts between components even when individual scripts work interactively.

## Decision

Maintain pytest coverage across major subsystems, including:

```text
synthetic generation
feature engineering
retrieval
agent tools
maintenance agent
API
```

Current checkpoint:

```text
41 tests passed
```

## Consequences

Refactoring can be checked against existing behavioral contracts before commits and deployment.

---

# ADR-031 — Do Not Present the System as Production-Ready

**Status:** Accepted

## Context

The project demonstrates production-style engineering patterns but operates entirely on synthetic data.

Using terms such as:

```text
production-ready aircraft maintenance AI
```

would overstate what has been validated.

## Decision

Describe AeroMaintain as:

```text
an end-to-end AI engineering portfolio system
```

and:

```text
a synthetic decision-support demonstration
```

rather than an operational aviation product.

## Consequences

Project documentation can still discuss:

- production-style architecture,
- serving boundaries,
- leakage controls,
- monitoring requirements,
- deployment patterns,

while clearly separating those engineering patterns from real operational validation.

---

# ADR-032 — Preserve Human Authority

**Status:** Accepted

## Context

Aircraft maintenance is safety-sensitive.

A predictive model or retrieval system should not be represented as an autonomous maintenance authority.

## Decision

The architecture terminates at:

```text
evidence-driven investigation
```

for human review.

It does not produce:

```text
airworthiness approval
return-to-service authorization
approved maintenance instructions
autonomous component replacement decisions
```

## Consequences

System language consistently uses concepts such as:

```text
risk
evidence
investigation
decision support
```

rather than implying autonomous maintenance control.

---

# ADR-033 — Prefer Reproducibility Over Unnecessary Complexity

**Status:** Accepted

## Context

Portfolio AI projects can accumulate technologies simply to increase the number of tools listed in the stack.

That can make the system harder to understand without improving the engineering result.

## Decision

Add infrastructure only when it serves a specific architectural purpose.

Examples:

```text
Databricks
→ lakehouse processing and ML experimentation

MLflow
→ experiment tracking

scikit-learn
→ predictive model and local retrieval components

FastAPI
→ programmatic serving interface

Streamlit
→ analyst-facing investigation interface
```

A paid LLM, vector database, message queue, or additional cloud service is not introduced merely for branding.

## Consequences

The architecture remains explainable end-to-end.

---

# ADR-034 — Maintain a Free-to-Run Core System

**Status:** Accepted

## Context

The portfolio demonstration should remain accessible without requiring paid API credentials.

## Decision

The core runtime uses:

```text
local scikit-learn inference
local Parquet data
local LSA retrieval
deterministic agent orchestration
```

No paid LLM API is required.

## Consequences

A developer can inspect and run the core AI workflow without external inference charges.

---

# ADR-035 — Keep Training and Runtime Contracts Explicit

**Status:** Accepted

## Context

One common source of ML deployment failure is mismatch between:

```text
training features
```

and:

```text
serving features
```

## Decision

Maintain an explicit feature contract in the persisted model bundle and reproduce the corresponding latest-component features in the local feature builder.

## Consequences

Serving logic can validate and order the feature set expected by the trained preprocessing pipeline.

This also makes training/serving skew easier to detect.

---

# Decision Summary

The major AeroMaintain decisions can be summarized as:

| Area | Decision |
|---|---|
| Data | Custom synthetic aviation environment |
| Lakehouse | Bronze / Silver / Gold |
| Target | Failure within future 30 days |
| Censoring | Remove incomplete future horizons |
| ML split | Purged temporal evaluation |
| Leakage | Exclude latent and future-derived signals |
| Model | Class-balanced Logistic Regression baseline |
| Model output | Uncalibrated risk score |
| Tracking | MLflow |
| Serving | Persisted local inference bundle |
| Retrieval baseline | TF-IDF |
| Default retrieval | TF-IDF + TruncatedSVD / LSA |
| Retrieval evaluation | Hit@1, Hit@3, MRR |
| Knowledge base | Synthetic project-owned documents |
| Agent | Deterministic evidence-driven orchestration |
| Agent output | Structured evidence contract |
| API | FastAPI |
| UI | Streamlit |
| UI execution | Direct core-agent invocation |
| Runtime data | Small synthetic assets tracked in Git |
| External LLM | Not required |
| Maintenance authority | Human remains the decision-maker |

---

# Guiding Principle

The architecture is ultimately based on one principle:

> **Complexity should be introduced only when it improves measurement, traceability, reproducibility, or the quality of the human decision-support workflow.**

AeroMaintain therefore favors explicit engineering contracts over hidden behavior and measurable components over unsupported AI claims.