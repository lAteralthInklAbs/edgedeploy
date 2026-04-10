"""Centralized configuration for EdgeDeploy."""
from pydantic_settings import BaseSettings
from pydantic import Field


class EdgeDeployConfig(BaseSettings):
    model_config = {"env_prefix": "EDGEDEPLOY_"}

    aws_region: str = "eu-central-1"
    s3_bucket: str = ""
    sagemaker_role: str = Field(default="", repr=False)

    # Quantization
    high_sensitivity_threshold: float = 0.05
    medium_sensitivity_threshold: float = 0.02
    qat_epochs: int = 10

    # Drift detection
    psi_threshold: float = 0.2
    mmd_threshold: float = 0.1
    adwin_confidence: float = 0.002
    persistence_windows: int = 3

    # Canary
    canary_stages: list[float] = [0.05, 0.50, 1.0]
    bake_time_minutes: int = 30


