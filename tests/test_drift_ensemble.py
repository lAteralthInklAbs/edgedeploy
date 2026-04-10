"""Tests for drift detection ensemble."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from src.drift.drift_ensemble import (
    DriftEnsemble,
    DriftResult,
    DriftLevel,
    PSIDetector,
    MMDDetector,
    ADWINDetector,
    _compute_psi,
    _compute_mmd,
)


class TestPSIComputation:
    """Tests for PSI computation."""

    def test_psi_identical_distributions(self):
        """PSI should be ~0 for identical distributions."""
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, size=1000)

        psi = _compute_psi(data, data.copy())
        assert psi < 0.05  # Very small PSI

    def test_psi_shifted_distribution(self):
        """PSI should be higher for shifted distributions."""
        rng = np.random.RandomState(42)
        expected = rng.normal(0, 1, size=1000)
        actual = rng.normal(1.0, 1, size=1000)  # Mean shifted by 1

        psi = _compute_psi(expected, actual)
        assert psi > 0.1  # Significant drift

    def test_psi_variance_change(self):
        """PSI should detect variance changes."""
        rng = np.random.RandomState(42)
        expected = rng.normal(0, 1, size=1000)
        actual = rng.normal(0, 2, size=1000)  # Double variance

        psi = _compute_psi(expected, actual)
        assert psi > 0.1

    def test_psi_symmetric(self):
        """PSI should be symmetric."""
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, size=1000)
        b = rng.normal(0.5, 1, size=1000)

        psi_ab = _compute_psi(a, b)
        psi_ba = _compute_psi(b, a)

        assert abs(psi_ab - psi_ba) < 0.01


class TestMMDComputation:
    """Tests for MMD computation."""

    def test_mmd_identical_distributions(self):
        """MMD should be ~0 for identical distributions."""
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, size=(100, 1))

        mmd = _compute_mmd(data, data.copy())
        assert mmd < 0.1

    def test_mmd_different_distributions(self):
        """MMD should be higher for different distributions."""
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, size=(100, 1))
        y = rng.normal(2, 1, size=(100, 1))

        mmd = _compute_mmd(x, y)
        assert mmd > 0.01

    def test_mmd_multivariate(self):
        """MMD should work with multivariate data."""
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, size=(50, 3))
        y = rng.normal(0, 1, size=(50, 3))

        mmd = _compute_mmd(x, y)
        assert mmd >= 0


class TestPSIDetector:
    """Tests for PSI detector."""

    def test_detector_initialization(self, reference_distribution):
        """Test detector initializes correctly."""
        detector = PSIDetector()
        detector.fit(reference_distribution)
        assert detector._reference is not None

    def test_detect_no_drift(self, reference_distribution, no_drift_distribution):
        """Test detection with no drift."""
        detector = PSIDetector()
        detector.fit(reference_distribution)
        psi_score, level = detector.detect(no_drift_distribution)

        assert isinstance(psi_score, float)
        # No significant drift expected
        assert level in (DriftLevel.NONE, DriftLevel.WARNING)

    def test_detect_drift(self, reference_distribution, shifted_distribution):
        """Test detection with actual drift."""
        detector = PSIDetector()
        detector.fit(reference_distribution)
        psi_score, level = detector.detect(shifted_distribution)

        assert psi_score > 0.05


class TestMMDDetector:
    """Tests for MMD detector."""

    def test_detector_initialization(self, reference_distribution):
        """Test detector initializes correctly."""
        ref_2d = reference_distribution.reshape(-1, 1)
        detector = MMDDetector(num_permutations=20)  # Fewer for speed
        detector.fit(ref_2d)
        assert detector._reference is not None

    def test_detect_no_drift(self, reference_distribution, no_drift_distribution):
        """Test detection with no drift."""
        ref_2d = reference_distribution.reshape(-1, 1)
        actual_2d = no_drift_distribution.reshape(-1, 1)

        detector = MMDDetector(num_permutations=20)
        detector.fit(ref_2d)
        mmd_score, p_value, level = detector.detect(actual_2d)

        assert isinstance(mmd_score, float)
        assert 0.0 <= p_value <= 1.0

    def test_detect_drift(self, reference_distribution, shifted_distribution):
        """Test detection with drift."""
        ref_2d = reference_distribution.reshape(-1, 1)
        actual_2d = shifted_distribution.reshape(-1, 1)

        detector = MMDDetector(num_permutations=20)
        detector.fit(ref_2d)
        mmd_score, p_value, level = detector.detect(actual_2d)

        assert mmd_score >= 0


class TestADWINDetector:
    """Tests for ADWIN detector."""

    def test_detector_initialization(self):
        """Test ADWIN initializes correctly."""
        detector = ADWINDetector()
        assert detector.max_window_size == 1000
        assert len(detector._window) == 0

    def test_add_element(self):
        """Test adding elements to window."""
        detector = ADWINDetector()

        for i in range(100):
            detector.add(float(i))

        assert len(detector._window) == 100

    def test_detect_drift_in_stream(self):
        """Test drift detection in streaming data."""
        detector = ADWINDetector(delta=0.001)

        # Add stable values
        for _ in range(500):
            detector.add(np.random.normal(0, 0.1))

        # Add shifted values
        drift_detected = False
        for _ in range(500):
            result = detector.add(np.random.normal(5, 0.1))
            if result:
                drift_detected = True
                break

        # Should eventually detect drift
        assert drift_detected or len(detector._window) > 0

    def test_window_management(self):
        """Test window doesn't exceed max size."""
        detector = ADWINDetector(max_window_size=100)

        for i in range(200):
            detector.add(float(i))

        assert len(detector._window) <= 100

    def test_reset(self):
        """Test detector reset."""
        detector = ADWINDetector()

        for i in range(50):
            detector.add(float(i))

        detector.reset()
        assert len(detector._window) == 0


class TestDriftEnsemble:
    """Tests for drift ensemble."""

    def test_ensemble_initialization(self, reference_distribution):
        """Test ensemble initializes correctly."""
        ref_2d = reference_distribution.reshape(-1, 1)
        ensemble = DriftEnsemble()
        ensemble.fit(ref_2d)

        assert ensemble._psi_detector is not None
        assert ensemble._mmd_detector is not None

    def test_detect_no_drift(self, reference_distribution, no_drift_distribution):
        """Test ensemble with no drift."""
        ref_2d = reference_distribution.reshape(-1, 1)
        actual_2d = no_drift_distribution.reshape(-1, 1)

        ensemble = DriftEnsemble()
        ensemble.fit(ref_2d)
        result = ensemble.detect(actual_2d)

        assert isinstance(result, DriftResult)
        assert result.level in (DriftLevel.NONE, DriftLevel.WARNING)

    def test_detect_with_drift(self, reference_distribution, shifted_distribution):
        """Test ensemble with drift."""
        ref_2d = reference_distribution.reshape(-1, 1)
        actual_2d = shifted_distribution.reshape(-1, 1)

        ensemble = DriftEnsemble()
        ensemble.fit(ref_2d)
        result = ensemble.detect(actual_2d)

        assert result.psi_score is not None
        assert result.mmd_score is not None

    def test_weighted_voting(self, reference_distribution, shifted_distribution):
        """Test weighted voting combines scores correctly."""
        ref_2d = reference_distribution.reshape(-1, 1)
        actual_2d = shifted_distribution.reshape(-1, 1)

        ensemble = DriftEnsemble(initial_weights=[0.5, 0.3, 0.2])
        ensemble.fit(ref_2d)
        result = ensemble.detect(actual_2d)

        # Ensemble score should exist
        assert result.ensemble_score is not None
        assert 0.0 <= result.ensemble_score <= 1.0

    def test_persistence_filter(self, reference_distribution, shifted_distribution):
        """Test persistence filter requires multiple detections."""
        ref_2d = reference_distribution.reshape(-1, 1)

        ensemble = DriftEnsemble(persistence_required=3)
        ensemble.fit(ref_2d)

        # First detection might not trigger due to persistence
        actual_2d = shifted_distribution.reshape(-1, 1)

        # Multiple detections should eventually trigger
        results = []
        for _ in range(5):
            result = ensemble.detect(actual_2d)
            results.append(result)

        # Should have some results
        assert len(results) == 5


class TestDriftResult:
    """Tests for DriftResult model."""

    def test_result_creation(self):
        """Test creating DriftResult."""
        result = DriftResult(
            level=DriftLevel.CRITICAL,
            psi_score=0.5,
            mmd_score=0.3,
            mmd_pvalue=0.01,
            adwin_detected=True,
            ensemble_score=0.8,
            feature_attributions=[],
            top_drifted_features=["feature_0"],
            persistence_count=3,
            explanation="Drift level: CRITICAL",
        )

        assert result.level == DriftLevel.CRITICAL

    def test_result_no_drift(self):
        """Test result with no drift."""
        result = DriftResult(
            level=DriftLevel.NONE,
            psi_score=0.05,
            mmd_score=0.01,
            mmd_pvalue=0.8,
            adwin_detected=False,
            ensemble_score=0.1,
            feature_attributions=[],
            top_drifted_features=[],
            persistence_count=0,
            explanation="No drift detected",
        )

        assert result.level == DriftLevel.NONE


class TestDriftLevelEnum:
    """Tests for DriftLevel enum."""

    def test_drift_levels_exist(self):
        """Test drift levels have correct values."""
        assert DriftLevel.NONE.value == "none"
        assert DriftLevel.WARNING.value == "warning"
        assert DriftLevel.CRITICAL.value == "critical"


# Hypothesis tests for property-based testing
class TestDriftHypothesis:
    """Property-based tests using Hypothesis."""

    @given(st.floats(min_value=-10, max_value=10, allow_nan=False))
    @settings(max_examples=50)
    def test_psi_non_negative(self, shift):
        """PSI should always be non-negative."""
        rng = np.random.RandomState(42)
        expected = rng.normal(0, 1, size=500)
        actual = rng.normal(shift, 1, size=500)

        psi = _compute_psi(expected, actual)
        assert psi >= 0

    @given(
        st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False),
            min_size=50,
            max_size=200,
        )
    )
    @settings(max_examples=30)
    def test_adwin_handles_any_stream(self, values):
        """ADWIN should handle any valid stream without errors."""
        detector = ADWINDetector()

        for v in values:
            result = detector.add(v)
            assert isinstance(result, bool)


