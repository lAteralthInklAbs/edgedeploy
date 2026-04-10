# ADR-002: Ensemble Drift Detection with Adaptive Weights

## Status
Accepted

## Context
Production ML models degrade when input data distribution shifts from training data. Single drift detection methods have blind spots. Automotive perception systems require reliable drift detection for safety.

## Decision
Implement an ensemble drift detector combining three complementary methods:

1. **PSI (Population Stability Index)**: Detects univariate distribution shifts
2. **MMD (Maximum Mean Discrepancy)**: Detects multivariate distribution shifts via kernel embedding
3. **ADWIN (Adaptive Windowing)**: Detects concept drift in streaming data

### Ensemble Architecture
```
Reference Data ──┬── PSI Detector ────┐
                 │                     │
                 ├── MMD Detector ─────┼── Weighted Voting ── Drift Alert
                 │                     │
                 └── ADWIN Detector ───┘
```

### Default Weights
- PSI: 0.40 (good for feature drift)
- MMD: 0.35 (good for correlation drift)
- ADWIN: 0.25 (good for streaming/concept drift)

### Persistence Filter
Require 3 consecutive drift detections before alerting (reduces false positives).

## Consequences

### Positive
- Covers multiple drift types (feature, correlation, concept)
- Custom implementations (no external drift library dependencies)
- Persistence filter reduces alert fatigue
- Adaptive weights can be tuned per deployment

### Negative
- Higher computational cost than single method
- Requires reference data storage
- MMD scales poorly with very high dimensions

### Risks Mitigated
- Missed drift leading to model degradation
- False positives causing unnecessary retraining
- Silent accuracy degradation in production

## Implementation Notes

### PSI Thresholds (industry standard)
- < 0.1: No drift
- 0.1 - 0.2: Slight drift (monitor)
- > 0.2: Significant drift (alert)

### MMD Computation
Using RBF kernel with median heuristic for bandwidth selection.

## Alternatives Considered

1. **KS Test Only**: Univariate only, misses correlation shifts
2. **River/Alibi-detect**: External dependencies, less control
3. **Evidently AI**: Heavy dependency, overkill for edge

## Related
- ADR-003: Canary Deployment Rollback
- ISO 26262 Part 8: Supporting Processes


