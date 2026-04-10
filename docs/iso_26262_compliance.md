# ISO 26262 Compliance Implementation

This document maps ISO 26262 functional safety requirements to EdgeDeploy implementation.

| ISO 26262 Part | Requirement | Implementation |
|---------------|-------------|----------------|
| Part 4-6 | Safety Requirements Specification | fixtures/safety_requirements.json with structured trace IDs (REQ-SAF-xxx) |
| Part 6-9 | Software Unit Testing | tests/ with coverage targets per ASIL level |
| Part 6-10 | Software Integration Testing | test_canary.py, test_onnx_exporter.py end-to-end validation |
| Part 8 | Supporting Processes (Traceability) | requirements_tracer.py — BFS bidirectional graph |
| Part 8-9 | Change Management | Change impact analysis via forward_trace() / backward_trace() |
| Part 9 | ASIL-Oriented Analysis | ASIL classification drives deployment thresholds in canary.py |

## ASIL-Driven Deployment Policies

| ASIL Level | Canary Bake Time | Min Health Score | Shadow Required | Manual Approval |
|-----------|-----------------|-----------------|----------------|----------------|
| ASIL-D | 60 minutes | 0.95 | Yes | Yes |
| ASIL-C | 45 minutes | 0.92 | Yes | Yes |
| ASIL-B | 30 minutes | 0.88 | Optional | No |
| ASIL-A | 15 minutes | 0.85 | No | No |

## Traceability Hard Block

Deployment is blocked (SafetyTraceGapError) if ANY safety requirement lacks a linked passing test.
This is enforced in requirements_tracer.py via validate_deployment_readiness().
