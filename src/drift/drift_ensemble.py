"""
Drift Detection Ensemble with PSI, MMD, and ADWIN.

★DEEP MODULE - All algorithms implemented from scratch:
- PSI (Population Stability Index): bin-based distribution comparison
- MMD (Maximum Mean Discrepancy): kernel-based distance with permutation test
- ADWIN (Adaptive Windowing): streaming mean-shift detection
- Adaptive weighted voting with detector performance tracking
- Temporal persistence filter (3 consecutive windows)
- Feature attribution for drift diagnosis
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

SEED = 42


class DriftLevel(str, Enum):
    """Drift severity levels."""

    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"


class FeatureAttribution(BaseModel):
    """Attribution of drift to specific features."""

    feature_index: int
    feature_name: str
    contribution: float = Field(ge=0.0, le=1.0)
    psi_value: float
    is_drifted: bool


class DriftResult(BaseModel):
    """Complete drift detection result."""

    level: DriftLevel
    psi_score: float
    mmd_score: float
    mmd_pvalue: float
    adwin_detected: bool
    ensemble_score: float
    feature_attributions: list[FeatureAttribution]
    top_drifted_features: list[str]
    persistence_count: int
    explanation: str


def _compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute Population Stability Index between two distributions.

    PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))

    Args:
        expected: Reference distribution
        actual: Current distribution
        num_bins: Number of bins for histogram
        epsilon: Small value to avoid log(0)

    Returns:
        PSI value (0 = identical, >0.25 = significant drift)
    """
    # Compute bin edges from expected distribution
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)

    # Compute histograms
    expected_hist, _ = np.histogram(expected, bins=bin_edges)
    actual_hist, _ = np.histogram(actual, bins=bin_edges)

    # Convert to percentages
    expected_pct = expected_hist / len(expected) + epsilon
    actual_pct = actual_hist / len(actual) + epsilon

    # Compute PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return float(psi)


def _rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Compute RBF (Gaussian) kernel between two vectors."""
    diff = x - y
    return float(np.exp(-np.sum(diff ** 2) / (2 * sigma ** 2)))


def _compute_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: float = 1.0,
) -> float:
    """
    Compute Maximum Mean Discrepancy between two samples.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

    Args:
        X: First sample (n x d)
        Y: Second sample (m x d)
        sigma: RBF kernel bandwidth

    Returns:
        MMD^2 value
    """
    n, m = len(X), len(Y)

    # E[k(x,x')]
    xx_sum = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                xx_sum += _rbf_kernel(X[i], X[j], sigma)
    xx_term = xx_sum / (n * (n - 1)) if n > 1 else 0.0

    # E[k(y,y')]
    yy_sum = 0.0
    for i in range(m):
        for j in range(m):
            if i != j:
                yy_sum += _rbf_kernel(Y[i], Y[j], sigma)
    yy_term = yy_sum / (m * (m - 1)) if m > 1 else 0.0

    # E[k(x,y)]
    xy_sum = 0.0
    for i in range(n):
        for j in range(m):
            xy_sum += _rbf_kernel(X[i], Y[j], sigma)
    xy_term = xy_sum / (n * m) if n > 0 and m > 0 else 0.0

    mmd_squared = xx_term + yy_term - 2 * xy_term
    return max(0.0, mmd_squared)


def _mmd_permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    num_permutations: int = 100,
    sigma: float = 1.0,
    seed: int = SEED,
) -> tuple[float, float]:
    """
    Perform permutation test for MMD significance.

    Args:
        X: First sample
        Y: Second sample
        num_permutations: Number of permutations
        sigma: RBF kernel bandwidth
        seed: Random seed

    Returns:
        Tuple of (observed_mmd, p_value)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Observed MMD
    observed_mmd = _compute_mmd(X, Y, sigma)

    # Combine samples
    combined = np.vstack([X, Y])
    n = len(X)

    # Permutation test
    count_greater = 0
    for _ in range(num_permutations):
        # Shuffle and split
        np.random.shuffle(combined)
        X_perm = combined[:n]
        Y_perm = combined[n:]

        perm_mmd = _compute_mmd(X_perm, Y_perm, sigma)
        if perm_mmd >= observed_mmd:
            count_greater += 1

    p_value = (count_greater + 1) / (num_permutations + 1)
    return observed_mmd, p_value


@dataclass
class ADWINDetector:
    """
    ADWIN (Adaptive Windowing) drift detector.

    Maintains two sub-windows and detects when their means differ significantly.
    """

    delta: float = 0.002  # Confidence parameter
    min_window_size: int = 10
    max_window_size: int = 1000

    _window: deque[float] = field(default_factory=deque)
    _sum: float = field(init=False, default=0.0)
    _variance: float = field(init=False, default=0.0)
    _width: int = field(init=False, default=0)

    def add(self, value: float) -> bool:
        """
        Add a value and check for drift.

        Args:
            value: New observation

        Returns:
            True if drift detected
        """
        self._window.append(value)
        self._sum += value
        self._width += 1

        # Limit window size
        while len(self._window) > self.max_window_size:
            removed = self._window.popleft()
            self._sum -= removed
            self._width -= 1

        # Check for drift by comparing sub-windows
        if self._width < 2 * self.min_window_size:
            return False

        return self._check_drift()

    def _check_drift(self) -> bool:
        """Check if there's a significant mean difference between sub-windows."""
        window_list = list(self._window)
        n = len(window_list)

        # Try different split points
        for split in range(self.min_window_size, n - self.min_window_size):
            w0 = window_list[:split]
            w1 = window_list[split:]

            n0, n1 = len(w0), len(w1)
            if n0 < self.min_window_size or n1 < self.min_window_size:
                continue

            # Compute means
            mean0 = sum(w0) / n0
            mean1 = sum(w1) / n1

            # Hoeffding bound for difference
            m = 1 / (1/n0 + 1/n1)
            epsilon = math.sqrt((1 / (2 * m)) * math.log(4 / self.delta))

            if abs(mean0 - mean1) > epsilon:
                # Drift detected - shrink window
                self._window = deque(w1)
                self._sum = sum(w1)
                self._width = len(w1)
                return True

        return False

    def reset(self) -> None:
        """Reset the detector."""
        self._window.clear()
        self._sum = 0.0
        self._width = 0


@dataclass
class PSIDetector:
    """PSI-based drift detector."""

    num_bins: int = 10
    warning_threshold: float = 0.1
    critical_threshold: float = 0.25

    _reference: np.ndarray | None = field(init=False, default=None)

    def fit(self, reference_data: np.ndarray) -> None:
        """Set reference distribution."""
        self._reference = reference_data.copy()

    def detect(self, current_data: np.ndarray) -> tuple[float, DriftLevel]:
        """
        Detect drift using PSI.

        Returns:
            Tuple of (psi_score, drift_level)
        """
        if self._reference is None:
            raise ValueError("Must call fit() first")

        psi = _compute_psi(self._reference, current_data, self.num_bins)

        if psi >= self.critical_threshold:
            return psi, DriftLevel.CRITICAL
        elif psi >= self.warning_threshold:
            return psi, DriftLevel.WARNING
        else:
            return psi, DriftLevel.NONE


@dataclass
class MMDDetector:
    """MMD-based drift detector with permutation test."""

    sigma: float = 1.0
    num_permutations: int = 100
    alpha: float = 0.05

    _reference: np.ndarray | None = field(init=False, default=None)

    def fit(self, reference_data: np.ndarray) -> None:
        """Set reference distribution."""
        self._reference = reference_data.copy()

    def detect(self, current_data: np.ndarray) -> tuple[float, float, DriftLevel]:
        """
        Detect drift using MMD with permutation test.

        Returns:
            Tuple of (mmd_score, p_value, drift_level)
        """
        if self._reference is None:
            raise ValueError("Must call fit() first")

        mmd, p_value = _mmd_permutation_test(
            self._reference, current_data, self.num_permutations, self.sigma
        )

        if p_value < self.alpha:
            return mmd, p_value, DriftLevel.CRITICAL
        elif p_value < self.alpha * 2:
            return mmd, p_value, DriftLevel.WARNING
        else:
            return mmd, p_value, DriftLevel.NONE


@dataclass
class DriftEnsemble:
    """
    Ensemble drift detector combining PSI, MMD, and ADWIN.

    Features:
    - Adaptive weighted voting based on detector performance
    - Temporal persistence filter (requires 3 consecutive detections)
    - Feature attribution for drift diagnosis
    """

    # Initial weights [PSI, MMD, ADWIN]
    initial_weights: list[float] = field(default_factory=lambda: [0.4, 0.35, 0.25])

    # Persistence filter
    persistence_required: int = 3

    # Detectors
    _psi_detector: PSIDetector = field(default_factory=PSIDetector)
    _mmd_detector: MMDDetector = field(default_factory=MMDDetector)
    _adwin_detectors: dict[int, ADWINDetector] = field(default_factory=dict)

    # Adaptive weights
    _weights: list[float] = field(init=False)
    _persistence_counter: int = field(init=False, default=0)
    _last_level: DriftLevel = field(init=False, default=DriftLevel.NONE)

    # Performance tracking
    _true_positives: list[int] = field(default_factory=lambda: [0, 0, 0])
    _false_positives: list[int] = field(default_factory=lambda: [0, 0, 0])

    def __post_init__(self) -> None:
        self._weights = list(self.initial_weights)

    def fit(self, reference_data: np.ndarray) -> None:
        """
        Fit detectors on reference data.

        Args:
            reference_data: Reference distribution (n_samples x n_features)
        """
        self._psi_detector.fit(reference_data)
        self._mmd_detector.fit(reference_data)

        # Initialize ADWIN for each feature
        n_features = reference_data.shape[1] if reference_data.ndim > 1 else 1
        self._adwin_detectors = {i: ADWINDetector() for i in range(n_features)}

        # Feed reference data to ADWIN
        if reference_data.ndim == 1:
            reference_data = reference_data.reshape(-1, 1)

        for i in range(n_features):
            for val in reference_data[:, i]:
                self._adwin_detectors[i].add(float(val))

    def detect(
        self,
        current_data: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> DriftResult:
        """
        Detect drift using ensemble of methods.

        Args:
            current_data: Current data (n_samples x n_features)
            feature_names: Optional feature names for attribution

        Returns:
            DriftResult with ensemble decision
        """
        if current_data.ndim == 1:
            current_data = current_data.reshape(-1, 1)

        n_features = current_data.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # PSI detection
        psi_scores = []
        for i in range(n_features):
            ref_col = self._psi_detector._reference[:, i] if self._psi_detector._reference is not None and self._psi_detector._reference.ndim > 1 else self._psi_detector._reference
            psi = _compute_psi(
                ref_col if ref_col is not None else current_data[:, i],
                current_data[:, i],
            )
            psi_scores.append(psi)
        psi_score = np.mean(psi_scores)
        psi_level = (
            DriftLevel.CRITICAL if psi_score >= 0.25
            else DriftLevel.WARNING if psi_score >= 0.1
            else DriftLevel.NONE
        )

        # MMD detection
        mmd_score, mmd_pvalue, mmd_level = self._mmd_detector.detect(current_data)

        # ADWIN detection
        adwin_detections = []
        for i in range(n_features):
            for val in current_data[:, i]:
                detected = self._adwin_detectors[i].add(float(val))
                if detected:
                    adwin_detections.append(i)

        adwin_detected = len(adwin_detections) > 0
        adwin_level = DriftLevel.CRITICAL if adwin_detected else DriftLevel.NONE

        # Weighted ensemble vote
        level_scores = {DriftLevel.NONE: 0, DriftLevel.WARNING: 1, DriftLevel.CRITICAL: 2}
        levels = [psi_level, mmd_level, adwin_level]
        weighted_score = sum(
            self._weights[i] * level_scores[levels[i]]
            for i in range(3)
        )
        ensemble_score = weighted_score / 2  # Normalize to 0-1

        # Determine ensemble level
        if ensemble_score >= 0.7:
            raw_level = DriftLevel.CRITICAL
        elif ensemble_score >= 0.3:
            raw_level = DriftLevel.WARNING
        else:
            raw_level = DriftLevel.NONE

        # Apply persistence filter
        if raw_level == self._last_level and raw_level != DriftLevel.NONE:
            self._persistence_counter += 1
        else:
            self._persistence_counter = 1 if raw_level != DriftLevel.NONE else 0

        self._last_level = raw_level

        # Only escalate to critical if persistence threshold met
        if self._persistence_counter >= self.persistence_required:
            final_level = raw_level
        elif self._persistence_counter > 0:
            final_level = DriftLevel.WARNING
        else:
            final_level = DriftLevel.NONE

        # Feature attribution
        attributions = []
        total_psi = sum(psi_scores) + 1e-10
        for i, (psi, name) in enumerate(zip(psi_scores, feature_names, strict=True)):
            attributions.append(
                FeatureAttribution(
                    feature_index=i,
                    feature_name=name,
                    contribution=psi / total_psi,
                    psi_value=psi,
                    is_drifted=psi >= 0.1 or i in adwin_detections,
                )
            )

        # Top drifted features
        attributions.sort(key=lambda a: a.contribution, reverse=True)
        top_drifted = [a.feature_name for a in attributions[:5] if a.is_drifted]

        # Generate explanation
        explanation = self._generate_explanation(
            final_level, psi_score, mmd_pvalue, adwin_detected, top_drifted
        )

        return DriftResult(
            level=final_level,
            psi_score=psi_score,
            mmd_score=mmd_score,
            mmd_pvalue=mmd_pvalue,
            adwin_detected=adwin_detected,
            ensemble_score=ensemble_score,
            feature_attributions=attributions,
            top_drifted_features=top_drifted,
            persistence_count=self._persistence_counter,
            explanation=explanation,
        )

    def _generate_explanation(
        self,
        level: DriftLevel,
        psi: float,
        mmd_pvalue: float,
        adwin: bool,
        top_features: list[str],
    ) -> str:
        """Generate human-readable drift explanation."""
        parts = [f"Drift level: {level.value.upper()}"]

        if level != DriftLevel.NONE:
            parts.append(f"PSI: {psi:.3f}")
            parts.append(f"MMD p-value: {mmd_pvalue:.3f}")
            if adwin:
                parts.append("ADWIN detected mean shift")
            if top_features:
                parts.append(f"Top drifted features: {', '.join(top_features[:3])}")

        return ". ".join(parts)

    def update_weights(self, detector_idx: int, was_correct: bool) -> None:
        """
        Update detector weights based on performance.

        Args:
            detector_idx: Index of detector (0=PSI, 1=MMD, 2=ADWIN)
            was_correct: Whether the detection was correct
        """
        if was_correct:
            self._true_positives[detector_idx] += 1
            # Increase weight for correct detector
            self._weights[detector_idx] *= 1.05
        else:
            self._false_positives[detector_idx] += 1
            # Decrease weight for incorrect detector
            self._weights[detector_idx] *= 0.95

        # Normalize weights
        total = sum(self._weights)
        self._weights = [w / total for w in self._weights]

    def reset(self) -> None:
        """Reset detector state."""
        self._persistence_counter = 0
        self._last_level = DriftLevel.NONE
        for detector in self._adwin_detectors.values():
            detector.reset()


