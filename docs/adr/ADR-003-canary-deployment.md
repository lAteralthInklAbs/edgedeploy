# ADR-003: Progressive Canary Deployment with Auto-Rollback

## Status
Accepted

## Context
Deploying ML models to automotive edge devices requires extreme caution. A bad deployment could impact vehicle safety systems. Rolling updates must be progressive with ability to quickly rollback.

## Decision
Implement multi-stage canary deployment with automatic rollback:

### Deployment Stages
1. **Canary (5%)**: Initial validation on small traffic slice
2. **Early Adopter (25%)**: Broader testing
3. **Majority (50%)**: Production validation
4. **Full Rollout (100%)**: Complete deployment

### Health Metrics Monitored
- Success rate (minimum: 95%)
- P99 latency (maximum: 500ms)
- Model accuracy (minimum: 90%)

### Rollback Triggers
- Success rate drops below threshold
- Latency exceeds threshold
- Accuracy degradation detected
- Manual trigger by operator

### Traffic Routing
Consistent hashing ensures same device always routes to same version during a stage (prevents A/B testing noise).

## Consequences

### Positive
- Limits blast radius of bad deployments
- Automatic detection and response to issues
- Progressive confidence building
- Full audit trail of deployment decisions

### Negative
- Longer total deployment time (hours vs minutes)
- Complexity in traffic routing logic
- Requires good observability infrastructure

### Risks Mitigated
- Widespread deployment of buggy model
- Undetected accuracy degradation
- Service outages during deployment

## Implementation Notes

### Stage Transition
- Each stage runs for configured duration
- Health checks run at configurable interval (default: 30s)
- Stage advances only if all health checks pass
- 3 consecutive failures triggers rollback

### Rollback Process
1. Stop advancing traffic percentage
2. Progressively reduce canary traffic to 0%
3. Log rollback reason and metrics
4. Alert operators

## Alternatives Considered

1. **Blue-Green**: All-or-nothing, higher risk
2. **Feature Flags**: Application-level, not model-level
3. **Shadow Mode**: Good for testing but not deployment

## Related
- ADR-002: Drift Detection
- ADR-004: Requirements Traceability
- ISO 26262 Part 7: Production and Operation


