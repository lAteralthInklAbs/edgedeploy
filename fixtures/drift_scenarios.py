"""Drift detection scenario fixtures with deterministic seed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SEED = 42


@dataclass
class DriftScenario:
    """A drift detection test scenario."""

    name: str
    description: str
    reference_data: np.ndarray
    actual_data: np.ndarray
    expected_drift: bool
    expected_level: str  # "none", "warning", "critical"
    metadata: dict[str, Any] | None = None


def create_no_drift_scenario(
    n_samples: int = 1000,
    n_features: int = 5,
    seed: int = SEED,
) -> DriftScenario:
    """Create scenario with no drift (same distribution)."""
    rng = np.random.RandomState(seed)

    reference = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    actual = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))

    return DriftScenario(
        name="no_drift",
        description="Reference and actual from same distribution",
        reference_data=reference,
        actual_data=actual,
        expected_drift=False,
        expected_level="none",
    )


def create_mean_shift_scenario(
    n_samples: int = 1000,
    n_features: int = 5,
    shift: float = 1.0,
    seed: int = SEED,
) -> DriftScenario:
    """Create scenario with mean shift drift."""
    rng = np.random.RandomState(seed)

    reference = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    actual = rng.normal(loc=shift, scale=1.0, size=(n_samples, n_features))

    expected_level = "critical" if shift >= 1.0 else "warning" if shift >= 0.5 else "none"

    return DriftScenario(
        name="mean_shift",
        description=f"Mean shifted by {shift} standard deviations",
        reference_data=reference,
        actual_data=actual,
        expected_drift=True,
        expected_level=expected_level,
        metadata={"shift_amount": shift},
    )


def create_variance_change_scenario(
    n_samples: int = 1000,
    n_features: int = 5,
    scale_factor: float = 2.0,
    seed: int = SEED,
) -> DriftScenario:
    """Create scenario with variance change drift."""
    rng = np.random.RandomState(seed)

    reference = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    actual = rng.normal(loc=0.0, scale=scale_factor, size=(n_samples, n_features))

    expected_level = "critical" if scale_factor >= 2.0 else "warning" if scale_factor >= 1.5 else "none"

    return DriftScenario(
        name="variance_change",
        description=f"Variance scaled by factor of {scale_factor}",
        reference_data=reference,
        actual_data=actual,
        expected_drift=True,
        expected_level=expected_level,
        metadata={"scale_factor": scale_factor},
    )


def create_gradual_drift_scenario(
    n_samples: int = 1000,
    n_features: int = 5,
    seed: int = SEED,
) -> DriftScenario:
    """Create scenario with gradual drift over time."""
    rng = np.random.RandomState(seed)

    reference = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))

    # Gradual shift: first half similar, second half shifted
    actual_first = rng.normal(loc=0.0, scale=1.0, size=(n_samples // 2, n_features))
    actual_second = rng.normal(loc=0.5, scale=1.0, size=(n_samples // 2, n_features))
    actual = np.vstack([actual_first, actual_second])

    return DriftScenario(
        name="gradual_drift",
        description="Gradual drift with shift in second half",
        reference_data=reference,
        actual_data=actual,
        expected_drift=True,
        expected_level="warning",
    )


def create_categorical_drift_scenario(
    n_samples: int = 1000,
    n_categories: int = 5,
    seed: int = SEED,
) -> DriftScenario:
    """Create scenario with categorical distribution drift."""
    rng = np.random.RandomState(seed)

    # Reference: uniform distribution
    ref_probs = np.ones(n_categories) / n_categories
    reference = rng.choice(n_categories, size=n_samples, p=ref_probs).reshape(-1, 1)

    # Actual: skewed distribution
    actual_probs = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
    actual_probs = actual_probs[:n_categories]
    actual_probs = actual_probs / actual_probs.sum()
    actual = rng.choice(n_categories, size=n_samples, p=actual_probs).reshape(-1, 1)

    return DriftScenario(
        name="categorical_drift",
        description="Categorical distribution shift",
        reference_data=reference.astype(float),
        actual_data=actual.astype(float),
        expected_drift=True,
        expected_level="critical",
    )


def create_multivariate_drift_scenario(
    n_samples: int = 1000,
    seed: int = SEED,
) -> DriftScenario:
    """Create scenario with multivariate correlation drift."""
    rng = np.random.RandomState(seed)

    # Reference: uncorrelated features
    reference = rng.normal(size=(n_samples, 3))

    # Actual: correlated features
    cov = np.array([
        [1.0, 0.8, 0.5],
        [0.8, 1.0, 0.6],
        [0.5, 0.6, 1.0],
    ])
    actual = rng.multivariate_normal([0, 0, 0], cov, size=n_samples)

    return DriftScenario(
        name="correlation_drift",
        description="Feature correlation structure changed",
        reference_data=reference,
        actual_data=actual,
        expected_drift=True,
        expected_level="warning",
    )


def create_streaming_drift_scenario(
    n_samples: int = 2000,
    drift_point: int = 1000,
    seed: int = SEED,
) -> DriftScenario:
    """Create scenario for streaming drift detection."""
    rng = np.random.RandomState(seed)

    # Reference (pre-drift)
    reference = rng.normal(loc=0.0, scale=1.0, size=(drift_point, 1))

    # Actual (post-drift with shift)
    actual = rng.normal(loc=1.5, scale=1.0, size=(n_samples - drift_point, 1))

    # Combined stream
    stream = np.vstack([reference, actual])

    return DriftScenario(
        name="streaming_drift",
        description=f"Concept drift at sample {drift_point}",
        reference_data=reference,
        actual_data=actual,
        expected_drift=True,
        expected_level="critical",
        metadata={
            "drift_point": drift_point,
            "full_stream": stream,
        },
    )


def get_all_scenarios() -> list[DriftScenario]:
    """Get all drift scenarios."""
    return [
        create_no_drift_scenario(),
        create_mean_shift_scenario(shift=0.3),
        create_mean_shift_scenario(shift=1.0),
        create_variance_change_scenario(scale_factor=1.5),
        create_variance_change_scenario(scale_factor=2.0),
        create_gradual_drift_scenario(),
        create_categorical_drift_scenario(),
        create_multivariate_drift_scenario(),
        create_streaming_drift_scenario(),
    ]


