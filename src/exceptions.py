"""Centralized exception hierarchy for EdgeDeploy."""

from src.safety.requirements_tracer import SafetyTraceGapError


class EdgeDeployError(Exception):
    def __init__(self, message: str, context: dict | None = None):
        self.context = context or {}
        super().__init__(message)


class RetriableError(EdgeDeployError):
    pass

class SageMakerThrottlingError(RetriableError):
    pass

class S3TransientError(RetriableError):
    pass


class NonRetriableError(EdgeDeployError):
    pass

class QuantizationAccuracyError(NonRetriableError):
    def __init__(self, actual: float, required: float, asil: str):
        super().__init__(
            f"Accuracy {actual:.3f} below {asil} threshold {required:.3f}",
            context={"actual": actual, "required": required, "asil": asil},
        )

class CanaryHealthError(NonRetriableError):
    pass

# Re-export from safety module
__all__ = [
    "EdgeDeployError", "RetriableError", "NonRetriableError",
    "SafetyTraceGapError", "SageMakerThrottlingError", "S3TransientError",
    "QuantizationAccuracyError", "CanaryHealthError",
]


