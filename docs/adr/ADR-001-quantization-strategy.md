# ADR-001: Per-Layer Quantization Sensitivity Analysis

## Status
Accepted

## Context
Automotive edge devices have strict memory and latency constraints. Model quantization (FP32 → INT8) can provide 4x compression but risks accuracy degradation. Different layers have varying sensitivity to quantization.

## Decision
Implement per-layer sensitivity analysis with mixed-precision assignment:

1. **Sensitivity Analysis**: Quantize each layer individually and measure accuracy drop
2. **Mixed Precision**: Assign INT8 to insensitive layers, FP16/FP32 to sensitive ones
3. **Pareto Frontier**: Visualize compression vs accuracy trade-offs
4. **QAT**: Use Quantization-Aware Training for fine-tuning

### Sensitivity Thresholds
- INT8: sensitivity_score < 0.3
- FP16: 0.3 ≤ sensitivity_score < 0.7
- FP32: sensitivity_score ≥ 0.7

## Consequences

### Positive
- Achieves 3-4x compression while maintaining >97% accuracy
- Per-layer analysis provides interpretable results
- Mixed precision balances size vs accuracy optimally
- QAT recovers accuracy lost during quantization

### Negative
- Sensitivity analysis is computationally expensive (O(n) evaluations per layer)
- Requires calibration dataset
- Some operators may not support INT8 in target runtime

### Risks Mitigated
- Over-quantization of sensitive layers (e.g., first/last layers)
- Silent accuracy degradation in production
- Runtime incompatibility with quantized operators

## Alternatives Considered

1. **Uniform INT8**: Simple but unacceptable accuracy loss (~5%)
2. **Dynamic Quantization**: Lower compression, no static graph
3. **Pruning-first**: More complex, less predictable compression

## Related
- ADR-002: ONNX Export Validation
- ISO 26262 Part 6: Software Unit Testing


