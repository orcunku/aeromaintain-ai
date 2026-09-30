# AeroMaintain AI — Evaluation Framework

## 1. Purpose

AeroMaintain AI evaluates its AI components as separate measurable systems rather than treating the final dashboard as proof that the underlying models work.

The current evaluation framework covers two primary AI capabilities:

1. predictive maintenance risk modeling,
2. maintenance-document retrieval.

The maintenance agent is then tested as an orchestration layer that combines those capabilities with maintenance history.

All reported results are produced using **synthetic project data and synthetic maintenance documentation**.

They should not be interpreted as evidence of performance on real aircraft, real maintenance records, OEM documentation, or operational aviation systems.

---

# 2. Evaluation Principles

The project follows several evaluation principles.

## 2.1 Preserve Time

Predictive maintenance is a forward-looking problem.

Model evaluation should therefore approximate the question:

> Can a model trained on earlier observations rank risk in later observations?

Randomly mixing observations across the synthetic year would not preserve this structure.

---

## 2.2 Prevent Target Leakage

Features must represent information available at prediction time.

Variables that directly or indirectly reveal future outcomes are excluded.

---

## 2.3 Handle Incomplete Future Horizons

An observation cannot receive a valid 30-day future label unless the dataset contains the complete future observation window.

This requires explicit right-censoring treatment.

---

## 2.4 Separate Model Selection from Final Testing

Validation data is used for:

- model comparison,
- threshold selection.

The held-out test period is evaluated only after the model and threshold have been selected.

---

## 2.5 Compare Retrieval Against a Baseline

Semantic retrieval is not assumed to be better simply because it is more sophisticated.

A lexical TF-IDF baseline is measured first.

The LSA retriever is then evaluated using the same benchmark.

---

## 2.6 Report Limitations With Results

Metrics are reported together with:

- class prevalence,
- synthetic-data status,
- benchmark size,
- calibration status,
- evaluation scope.

This reduces the risk of presenting isolated metrics without context.

---

# 3. Predictive Task

The predictive maintenance task is binary classification over a future time horizon.

For an observation at time `t`, the target asks:

```text
Does this component experience a failure event
within the next 30 days?
```

Conceptually:

```text
Features at time t
        │
        ▼
Future interval:
(t, t + 30 days]
        │
        ▼
Any failure?
   │          │
  yes         no
   │          │
target = 1  target = 0
```

The model is therefore intended to rank **future component risk**, not identify whether a failure is occurring on the current telemetry row.

---

# 4. Target Construction

The future target is constructed at component level using the component's subsequent observations.

The target horizon is:

```text
30 days
```

A positive observation indicates that at least one failure event occurs within that future interval.

The current row's `failure_event` is not used as a production predictive feature.

Future failure counts and other future-derived fields are also excluded from the model feature set.

---

# 5. Right-Censoring Policy

## 5.1 Problem

Suppose the dataset ends on:

```text
2025-12-31
```

An observation from:

```text
2025-12-20
```

does not contain 30 days of observable future data.

The absence of a recorded failure before the dataset ends does not establish that no failure would occur during the complete prediction horizon.

Treating this observation as a negative example would introduce incorrect target information.

## 5.2 Decision

Observations without a complete 30-day future horizon are excluded from supervised model training and evaluation.

After applying the future-horizon eligibility rule:

```text
Eligible observations: 167,500
Positive observations:   4,770
Positive rate:            ~2.85%
```

## 5.3 Interpretation

The removed rows are not assumed to be failures or non-failures.

Their future outcome for the complete prediction horizon is treated as unavailable.

---

# 6. Leakage Controls

Predictive evaluation is meaningful only if model inputs represent information that could be available at inference time.

The project therefore excludes several categories of leakage.

---

## 6.1 Latent Simulation State

The synthetic generator contains:

```text
health_index
```

as an internal degradation signal.

Although useful for generating synthetic behavior, it represents simulator state rather than a production-observable model feature.

It is excluded.

---

## 6.2 Hidden Susceptibility

Component susceptibility influences simulated degradation and failure behavior.

It is deliberately hidden from the predictive model.

This prevents the model from directly observing one of the variables used to create the synthetic outcome process.

---

## 6.3 Current Failure Indicator

The current telemetry field:

```text
failure_event
```

is excluded from the production feature set.

The task is future risk prediction rather than same-row failure detection.

---

## 6.4 Future Information

Future failure counts and future-derived target metadata are excluded.

---

## 6.5 Full-Dataset Outlier Information

The Silver pipeline contains a vibration outlier flag useful for data-quality analysis.

Its current threshold is calculated using broader dataset information.

The flag is therefore excluded from production ML features to avoid introducing a full-year-derived signal into historical predictions.

---

# 7. Production Feature Set

The selected model uses 20 production features.

## Categorical

```text
aircraft_type
component_type
```

## Numeric

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

Preprocessing is fitted using training data only.

---

# 8. Temporal Evaluation Design

AeroMaintain does not use a random train/test split for the predictive task.

The observations are divided chronologically.

```text
┌─────────────────────────────────────────────┐
│ TRAIN                                       │
│ through 2025-08-01                          │
│                                             │
│ 106,500 observations                        │
│ 2,648 positives                             │
│ positive rate ≈ 2.49%                       │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
               30-DAY PURGE GAP
                      │
                      ▼
┌─────────────────────────────────────────────┐
│ VALIDATION                                  │
│ 2025-09-01 → 2025-09-30                    │
│                                             │
│ 15,000 observations                         │
│ 400 positives                               │
│ positive rate ≈ 2.67%                       │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
               30-DAY PURGE GAP
                      │
                      ▼
┌─────────────────────────────────────────────┐
│ TEST                                        │
│ 2025-10-31 → 2025-12-01                    │
│                                             │
│ 16,000 observations                         │
│ 629 positives                               │
│ positive rate ≈ 3.93%                       │
└─────────────────────────────────────────────┘
```

---

# 9. Why Purge Gaps Are Used

The prediction target itself looks 30 days into the future.

Without a purge interval, observations on opposite sides of a dataset boundary could have overlapping future-label windows.

Example:

```text
Training observation
August 30
      │
      └──── future target window ────► September

Validation observation
September 1
```

Even though the feature timestamps belong to different splits, their outcome windows can overlap.

A 30-day purge gap is therefore introduced between supervised partitions.

This reduces temporal contamination between target windows.

---

# 10. Preprocessing Evaluation Boundary

Numeric preprocessing uses:

```text
median imputation
standard scaling
```

Categorical preprocessing uses:

```text
most-frequent imputation
one-hot encoding
```

The preprocessing pipeline is fitted on the training partition.

Validation and test data are transformed using the fitted training pipeline rather than independently learning preprocessing statistics.

This preserves the evaluation boundary.

---

# 11. Candidate Models

Two baseline model families were evaluated.

## 11.1 Class-Balanced Logistic Regression

Logistic Regression provides:

- a lightweight baseline,
- nonlinear behavior through engineered features and categorical expansion,
- simple persistence,
- efficient local inference.

Class balancing is used because the positive class is uncommon.

---

## 11.2 Random Forest

Random Forest provides a nonlinear tree-based comparison against the linear baseline.

The goal was not to exhaustively tune every possible estimator, but to determine whether additional model complexity improved the measured validation performance enough to justify selection.

---

# 12. Validation Results

The validation period contains:

```text
15,000 observations
400 positives
positive prevalence ≈ 0.0267
```

The positive prevalence is also the approximate expected PR-AUC of a random ranking.

## 12.1 PR-AUC

```text
Logistic Regression    0.0496
Random Forest          0.0353
Random baseline        0.0267
```

## 12.2 Best Validation F1

```text
Logistic Regression    0.1168
Random Forest          0.0889
```

## 12.3 Comparison

| Model | Validation PR-AUC | Best Validation F1 |
|---|---:|---:|
| Logistic Regression | **0.0496** | **0.1168** |
| Random Forest | 0.0353 | 0.0889 |
| Random PR baseline | 0.0267 | — |

The Logistic Regression model was selected for final evaluation.

---

# 13. Threshold Evaluation

A classification threshold of `0.5` was evaluated but was not automatically assumed to be the correct operating threshold.

At threshold:

```text
0.5
```

the Logistic Regression validation results were approximately:

```text
Precision    0.0374
Recall       0.7700
F1           0.0714
```

A threshold search on the validation set identified an F1-oriented threshold of approximately:

```text
0.7248
```

At that validation-selected threshold:

```text
Precision    0.0691
Recall       0.3775
F1           0.1168
```

The threshold was selected before evaluating the held-out test period.

---

# 14. Why the Threshold Is Not a Universal Maintenance Boundary

The selected threshold is an experimental decision threshold for the synthetic benchmark.

It is not:

```text
an OEM threshold
a regulatory threshold
an airworthiness threshold
a universal failure-risk boundary
```

In a real decision-support system, threshold selection would depend on factors such as:

- failure consequence,
- false-negative cost,
- false-positive inspection burden,
- component category,
- maintenance capacity,
- probability calibration,
- operational policy.

The current threshold exists to make the synthetic evaluation and downstream application behavior explicit.

---

# 15. Held-Out Test Evaluation

After model and threshold selection, the final Logistic Regression pipeline was evaluated on the later held-out test period.

Test dataset:

```text
16,000 observations
629 positives
positive prevalence ≈ 0.0393
```

---

# 16. Test Metrics

The held-out test results are:

| Metric | Result |
|---|---:|
| PR-AUC | **0.0713** |
| ROC-AUC | **0.6811** |
| Precision | **0.0802** |
| Recall | **0.4006** |
| F1 | **0.1337** |
| Positive prevalence | 0.0393 |

The random-ranking PR-AUC baseline is approximately equal to positive prevalence:

```text
0.0393
```

The observed PR-AUC lift is therefore approximately:

```text
0.0713 / 0.0393 ≈ 1.81x
```

This is reported as:

```text
~1.81x PR-AUC lift over the random baseline
```

for this synthetic held-out test period.

---

# 17. Test Confusion Matrix

Using the validation-selected threshold:

```text
                 Predicted
               Negative Positive

Actual Negative  12,481   2,890
Actual Positive     377     252
```

Equivalent counts:

```text
True negatives:   12,481
False positives:   2,890
False negatives:     377
True positives:      252
```

---

# 18. Interpreting Predictive Results

The results demonstrate that the selected model extracts some predictive ranking signal from the synthetic environment.

They do **not** demonstrate that the model is suitable for operational aircraft maintenance.

Several observations are important.

## 18.1 Precision Is Low

At the selected threshold:

```text
precision ≈ 0.0802
```

Many above-threshold observations are therefore false positives.

This would matter substantially in a real maintenance environment because unnecessary investigations have operational cost.

## 18.2 Recall Is Partial

The model identifies approximately:

```text
40.1%
```

of positive test observations at the selected threshold.

A substantial fraction of future positive observations remain below threshold.

## 18.3 Ranking Is Better Than Random

Test PR-AUC:

```text
0.0713
```

is above test positive prevalence:

```text
0.0393
```

but the absolute performance remains modest.

## 18.4 Synthetic Structure Matters

The model is learning relationships created by the project simulator.

Real aircraft component degradation may have different distributions, failure mechanisms, sensor behavior, maintenance interventions, and censoring processes.

---

# 19. Risk Score Calibration

The selected Logistic Regression model uses class balancing.

Its `predict_proba` output has not been validated through a separate probability-calibration procedure.

The application therefore labels the model output as:

```text
risk_score
```

and not:

```text
failure_probability
```

The inference contract explicitly records:

```text
calibrated_probability = False
```

For example:

```text
risk_score = 0.7145
```

must not be interpreted as:

```text
71.45% probability of failure within 30 days
```

---

# 20. Persisted Model Validation

The fitted inference pipeline is persisted as a joblib bundle.

After serialization, the artifact was reloaded and inference output was compared with the original model.

On the tested 100-row sample:

```text
maximum prediction difference = 0.0
```

This verifies serialization consistency for that test.

It does not independently validate predictive performance.

---

# 21. Retrieval Evaluation Objective

The retrieval subsystem has a separate evaluation problem.

Given an investigation query, the retriever should rank maintenance knowledge chunks relevant to the expected component/document context.

The retrieval benchmark is independent from the predictive ML evaluation.

---

# 22. Retrieval Corpus

The synthetic knowledge base contains documents for:

```text
HYDRAULIC_PUMP
GENERATOR
AIR_CYCLE_MACHINE
FUEL_PUMP
ACTUATOR
```

The ingestion pipeline currently produces:

```text
30 chunks
```

All documents are synthetic project content.

No OEM maintenance manual is used.

---

# 23. Retrieval Benchmark

The repository contains a fixed evaluation set:

```text
evaluation/retrieval_queries.json
```

Current benchmark size:

```text
10 queries
```

Each query contains expected relevance information used to evaluate ranked retrieval results.

The benchmark is intentionally small and exists to provide a repeatable comparison between project retrieval approaches.

---

# 24. Retrieval Metrics

Three metrics are used.

## 24.1 Hit@1

Hit@1 asks:

> Did the expected relevant result appear in the first retrieved position?

Conceptually:

```text
relevant result ranked #1
        │
        ▼
      success
```

---

## 24.2 Hit@3

Hit@3 asks:

> Did the expected relevant result appear anywhere in the top three retrieved results?

This is useful when the downstream agent consumes multiple evidence chunks.

---

## 24.3 Mean Reciprocal Rank

MRR rewards relevant results appearing earlier in the ranking.

For a query:

```text
reciprocal rank = 1 / rank_of_first_relevant_result
```

Examples:

```text
rank 1 → 1.00
rank 2 → 0.50
rank 3 → 0.33
```

MRR averages this value across the evaluation queries.

---

# 25. TF-IDF Retrieval Baseline

The first retrieval system uses TF-IDF lexical similarity.

Measured benchmark results:

```text
Hit@1    0.500
Hit@3    0.800
MRR      0.633
```

Interpretation on the 10-query benchmark:

```text
50% of queries had the expected relevant result at rank 1

80% had it somewhere within the top 3
```

---

# 26. LSA Retrieval Evaluation

The semantic retrieval implementation applies Truncated SVD to the TF-IDF representation.

Conceptually:

```text
Documents
   │
   ▼
TF-IDF Matrix
   │
   ▼
TruncatedSVD
   │
   ▼
Latent Semantic Space
   │
   ▼
Similarity Ranking
```

Measured benchmark results:

```text
Hit@1    0.700
Hit@3    1.000
MRR      0.850
```

---

# 27. Retrieval Comparison

| Retriever | Hit@1 | Hit@3 | MRR |
|---|---:|---:|---:|
| TF-IDF | 0.500 | 0.800 | 0.633 |
| **LSA** | **0.700** | **1.000** | **0.850** |

Absolute changes on the project benchmark:

```text
Hit@1    +0.200
Hit@3    +0.200
MRR      +0.217
```

Because LSA improved all three measured metrics, it became the default retrieval strategy used by the maintenance agent.

---

# 28. Retrieval Result Limitations

The LSA result should be interpreted carefully.

The benchmark contains only:

```text
10 synthetic queries
```

and the corpus contains:

```text
synthetic project-owned maintenance documents
```

Therefore the result demonstrates:

> LSA outperformed the TF-IDF baseline on the fixed AeroMaintain synthetic retrieval benchmark.

It does not establish:

```text
performance on OEM manuals
performance on regulatory documents
performance on large technical corpora
performance on real maintenance queries
```

A production retrieval evaluation would require a substantially larger expert-labeled benchmark.

---

# 29. Why Transformer Embeddings Are Not the Current Baseline

Transformer sentence embeddings were considered during development.

The current evaluation system instead uses LSA because it provides:

- fully local execution,
- no external model API,
- no runtime model download,
- low computational overhead,
- reproducible evaluation,
- measurable improvement over TF-IDF on the current benchmark.

This is an engineering trade-off, not a claim that LSA is universally superior to modern embedding models.

Future evaluation could compare:

```text
TF-IDF
LSA
local transformer embeddings
domain-specific embeddings
hybrid lexical + semantic retrieval
```

using the same expanded benchmark.

---

# 30. Agent Evaluation

The maintenance agent is deterministic rather than generative.

Its evaluation therefore focuses primarily on workflow correctness and contract behavior rather than language-generation quality.

The agent must correctly orchestrate:

```text
component validation
risk inference
maintenance-history retrieval
component-type resolution
query construction
evidence retrieval
investigation assembly
limitations
```

---

# 31. Agent Output Contract

The expected investigation structure contains:

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

Tests verify the behavior of this structured workflow.

This makes the agent easier to evaluate than an unconstrained natural-language-only response.

---

# 32. Example Investigation Check

A known project example is:

```text
CMP-00196
```

Observed predictive output:

```text
risk score:
~0.7145

selected threshold:
~0.7248

threshold status:
below threshold

prediction horizon:
30 days

calibrated probability:
False
```

The component resolves to:

```text
HYDRAULIC_PUMP
```

The maintenance-history workflow returns one event for the current synthetic dataset.

The evidence retrieval stage returns component-specific synthetic maintenance passages.

This example is useful as an integration check across:

```text
feature generation
→ prediction
→ maintenance history
→ retrieval
→ agent assembly
```

It is not a special-case rule encoded into the agent.

---

# 33. API Evaluation

FastAPI behavior is covered through automated tests with mocked or controlled dependencies where appropriate.

The API exposes:

```text
GET /
GET /health
GET /investigate/{component_id}
GET /components/{component_id}/risk
```

Tests validate the service contract without requiring every API test to rerun expensive underlying workflows.

---

# 34. Automated Test Suite

At the current project checkpoint:

```text
41 tests passed
```

The suite covers major areas including:

```text
synthetic data generation
feature engineering
retrieval
maintenance tools
maintenance agent
FastAPI
```

The test suite serves a different purpose from ML evaluation.

```text
pytest
→ verifies software behavior and contracts

ML / retrieval evaluation
→ measures predictive or ranking quality
```

Passing software tests therefore does not imply high model accuracy, and model metrics do not replace software tests.

---

# 35. Evaluation Layers

The complete project uses several different forms of validation.

```text
┌───────────────────────────────────────────────┐
│ SOFTWARE TESTING                              │
│                                               │
│ Does the implementation behave as expected?  │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│ PREDICTIVE ML EVALUATION                      │
│                                               │
│ Does the model rank future synthetic risk?    │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│ RETRIEVAL EVALUATION                          │
│                                               │
│ Does retrieval rank expected evidence well?   │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│ INTEGRATION CHECK                             │
│                                               │
│ Do model + history + retrieval + agent work?  │
└───────────────────────────────────────────────┘
```

These evaluation layers should not be collapsed into a single metric.

---

# 36. What Has Not Been Evaluated

The current project has not established:

- real-world aircraft predictive performance,
- real component failure probabilities,
- model calibration for operational risk communication,
- performance across different airlines or fleets,
- robustness to real sensor drift,
- robustness to changing maintenance policies,
- expert-rated retrieval quality on OEM documentation,
- human-factors performance,
- operational maintenance cost reduction,
- safety impact,
- regulatory compliance,
- production service reliability.

These require real operational data, domain experts, controlled validation, and appropriate governance.

---

# 37. Future Predictive Evaluation

Potential extensions include:

## Probability Calibration

Evaluate:

```text
Brier score
reliability diagrams
calibration error
```

using an appropriately separated calibration dataset.

## Cost-Sensitive Evaluation

Model the different costs of:

```text
false negatives
false positives
inspection workload
missed failures
```

rather than selecting a threshold using F1 alone.

## Time-to-Event Modeling

Compare binary horizon classification with:

```text
survival analysis
hazard modeling
remaining-useful-life approaches
```

## Stronger Predictive Models

Evaluate models such as:

```text
gradient boosting
calibrated tree ensembles
temporal models
```

while preserving the same temporal leakage controls.

---

# 38. Future Retrieval Evaluation

A stronger retrieval benchmark would include:

```text
more queries
multiple relevant passages per query
hard negative documents
ambiguous component symptoms
cross-component terminology
expert relevance judgments
```

Potential metrics could include:

```text
Recall@K
Precision@K
nDCG
MAP
MRR
```

The same benchmark could then compare:

```text
TF-IDF
LSA
dense embeddings
hybrid retrieval
reranking
```

---

# 39. Future Agent Evaluation

If a generative model is later added to AeroMaintain, evaluation should remain grounded in the structured evidence contract.

Potential criteria include:

```text
evidence faithfulness
citation correctness
unsupported-claim rate
component consistency
risk-score preservation
maintenance-history preservation
instruction safety
```

The generative layer should be evaluated separately from retrieval quality and predictive model quality.

---

# 40. Reproducibility

The repository retains the components required to reproduce the local demonstration:

```text
synthetic runtime datasets
persisted predictive model
synthetic knowledge documents
retrieval benchmark
evaluation code
automated tests
```

Training infrastructure is separated from serving infrastructure.

Databricks is used for the lakehouse and model-development workflow, while the persisted application runtime can execute locally.

---

# 41. Evaluation Summary

Current measured project results:

## Predictive ML

```text
Model:
Class-balanced Logistic Regression

Prediction horizon:
30 days

Evaluation:
Purged temporal split

Test PR-AUC:
0.0713

Test random PR baseline:
0.0393

PR-AUC lift:
~1.81x

Test ROC-AUC:
0.6811

Test precision:
0.0802

Test recall:
0.4006

Test F1:
0.1337

Probability calibrated:
No
```

## Retrieval

```text
Benchmark:
10 synthetic queries

TF-IDF:
Hit@1 = 0.500
Hit@3 = 0.800
MRR   = 0.633

LSA:
Hit@1 = 0.700
Hit@3 = 1.000
MRR   = 0.850

Selected retriever:
LSA
```

## Software

```text
Automated tests:
41 passed
```

---

# 42. Final Interpretation

AeroMaintain's evaluation framework is designed to demonstrate disciplined AI engineering rather than maximize a single metric.

The predictive model shows measurable but modest risk-ranking signal on held-out synthetic future data.

The LSA retriever improves over the project's TF-IDF baseline on a small synthetic retrieval benchmark.

The deterministic agent then combines these independently inspectable capabilities with maintenance history.

The most important evaluation principle is therefore:

> **Every AI capability should have an explicit task, an explicit measurement method, and an explicit boundary on what its results prove.**