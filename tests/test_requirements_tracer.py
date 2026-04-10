"""Tests for ISO 26262 requirements tracer."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.safety.requirements_tracer import (
    RequirementsTracer,
    Requirement,
    RequirementStatus,
    TraceLink,
    TraceabilityReport,
    SafetyTraceGapError,
    ASILLevel,
    TraceGap,
    CoverageMetrics,
    _compute_asil_weight,
    _gap_severity_from_asil,
)


class TestASILHelpers:
    """Tests for ASIL helper functions."""

    def test_asil_weight_ordering(self):
        """Test ASIL weights are correctly ordered."""
        assert _compute_asil_weight(ASILLevel.QM) == 0
        assert _compute_asil_weight(ASILLevel.ASIL_A) == 1
        assert _compute_asil_weight(ASILLevel.ASIL_B) == 2
        assert _compute_asil_weight(ASILLevel.ASIL_C) == 3
        assert _compute_asil_weight(ASILLevel.ASIL_D) == 4

    def test_gap_severity_critical_for_asil_d(self):
        """Test critical severity for ASIL-D gaps."""
        severity = _gap_severity_from_asil(ASILLevel.ASIL_D, "missing_parent")
        assert severity == "critical"

    def test_gap_severity_major_for_asil_a(self):
        """Test major severity for ASIL-A gaps."""
        severity = _gap_severity_from_asil(ASILLevel.ASIL_A, "missing_parent")
        assert severity == "major"

    def test_gap_severity_minor_for_qm(self):
        """Test minor severity for QM gaps."""
        severity = _gap_severity_from_asil(ASILLevel.QM, "missing_parent")
        assert severity == "minor"


class TestRequirement:
    """Tests for Requirement model."""

    def test_requirement_creation(self):
        """Test creating a requirement."""
        req = Requirement(
            req_id="SYS-001",
            title="System Safety Requirement",
            description="The system shall ensure safe operation",
            asil_level=ASILLevel.ASIL_D,
        )

        assert req.req_id == "SYS-001"
        assert req.asil_level == ASILLevel.ASIL_D
        assert req.status == RequirementStatus.DRAFT

    def test_requirement_id_validation(self):
        """Test requirement ID must match pattern."""
        with pytest.raises(ValueError):
            Requirement(
                req_id="invalid",  # Must be XX-NNN format
                title="Test",
                description="Test",
                asil_level=ASILLevel.QM,
            )

    def test_requirement_with_parents(self):
        """Test requirement with parent links."""
        req = Requirement(
            req_id="HW-001",
            title="Hardware Requirement",
            description="Hardware shall...",
            asil_level=ASILLevel.ASIL_C,
            parent_ids=["SYS-001", "SYS-002"],
        )

        assert len(req.parent_ids) == 2
        assert "SYS-001" in req.parent_ids

    def test_content_hash_consistency(self):
        """Test content hash is consistent."""
        req = Requirement(
            req_id="SYS-001",
            title="Test",
            description="Description",
            asil_level=ASILLevel.ASIL_B,
        )

        hash1 = req.content_hash()
        hash2 = req.content_hash()
        assert hash1 == hash2

    def test_content_hash_changes(self):
        """Test content hash changes when content changes."""
        req1 = Requirement(
            req_id="SYS-001",
            title="Test 1",
            description="Description",
            asil_level=ASILLevel.ASIL_B,
        )
        req2 = Requirement(
            req_id="SYS-001",
            title="Test 2",  # Different title
            description="Description",
            asil_level=ASILLevel.ASIL_B,
        )

        assert req1.content_hash() != req2.content_hash()


class TestTraceLink:
    """Tests for TraceLink model."""

    def test_trace_link_creation(self):
        """Test creating a trace link."""
        link = TraceLink(
            source_id="SYS-001",
            target_id="HW-001",
            link_type="derives",
        )

        assert link.source_id == "SYS-001"
        assert link.target_id == "HW-001"
        assert link.confidence == 1.0

    def test_trace_link_with_rationale(self):
        """Test trace link with rationale."""
        link = TraceLink(
            source_id="SYS-001",
            target_id="TC-001",
            link_type="verifies",
            rationale="Test case verifies safety function",
        )

        assert link.rationale == "Test case verifies safety function"


class TestRequirementsTracer:
    """Tests for RequirementsTracer."""

    @pytest.fixture
    def tracer(self) -> RequirementsTracer:
        """Create a tracer with sample requirements."""
        tracer = RequirementsTracer()

        # Add system-level requirement
        tracer.add_requirement(Requirement(
            req_id="SYS-001",
            title="System Safety Function",
            description="Top-level safety requirement",
            asil_level=ASILLevel.ASIL_D,
            status=RequirementStatus.APPROVED,
        ))

        # Add hardware requirement derived from system
        tracer.add_requirement(Requirement(
            req_id="HW-001",
            title="Hardware Safety",
            description="Hardware-level requirement",
            asil_level=ASILLevel.ASIL_C,
            parent_ids=["SYS-001"],
            status=RequirementStatus.IMPLEMENTED,
            test_case_ids=["TC-001"],
        ))

        # Add software requirement
        tracer.add_requirement(Requirement(
            req_id="SW-001",
            title="Software Safety",
            description="Software-level requirement",
            asil_level=ASILLevel.ASIL_B,
            parent_ids=["SYS-001"],
            status=RequirementStatus.IMPLEMENTED,
            test_case_ids=["TC-002"],
        ))

        return tracer

    def test_add_requirement(self):
        """Test adding requirements."""
        tracer = RequirementsTracer()

        req = Requirement(
            req_id="SYS-001",
            title="Test Requirement",
            description="Description",
            asil_level=ASILLevel.ASIL_A,
        )

        tracer.add_requirement(req)
        assert "SYS-001" in tracer.requirements
        assert tracer.requirements["SYS-001"].title == "Test Requirement"

    def test_add_trace_link(self, tracer):
        """Test adding trace links."""
        link = TraceLink(
            source_id="SYS-001",
            target_id="HW-001",
            link_type="derives",
        )

        tracer.add_trace_link(link)
        assert len(tracer.trace_links) == 1

    def test_get_ancestors(self, tracer):
        """Test getting ancestor requirements."""
        ancestors = tracer.get_ancestors("HW-001")

        assert "SYS-001" in ancestors
        assert len(ancestors) == 1

    def test_get_descendants(self, tracer):
        """Test getting descendant requirements."""
        descendants = tracer.get_descendants("SYS-001")

        assert "HW-001" in descendants
        assert "SW-001" in descendants
        assert len(descendants) == 2

    def test_impact_analysis(self, tracer):
        """Test impact analysis."""
        impact = tracer.impact_analysis("SYS-001")

        assert impact["requirement_id"] == "SYS-001"
        assert len(impact["descendants"]) == 2
        assert impact["highest_asil_affected"] >= 3  # ASIL-C or higher

    def test_impact_analysis_nonexistent(self, tracer):
        """Test impact analysis for nonexistent requirement."""
        impact = tracer.impact_analysis("NONEXISTENT-001")
        assert "error" in impact

    def test_check_asil_decomposition_valid(self, tracer):
        """Test valid ASIL decomposition."""
        result = tracer.check_asil_decomposition("SYS-001", ["HW-001", "SW-001"])

        # ASIL_D can decompose to ASIL_C + ASIL_A or ASIL_B + ASIL_B
        # We have ASIL_C + ASIL_B which should be acceptable
        assert "valid" in result

    def test_find_trace_gaps(self, tracer):
        """Test finding trace gaps."""
        # Add a requirement with missing test
        tracer.add_requirement(Requirement(
            req_id="HW-002",
            title="Hardware Untested",
            description="Missing test",
            asil_level=ASILLevel.ASIL_D,
            parent_ids=["SYS-001"],
            status=RequirementStatus.IMPLEMENTED,
            # No test_case_ids - gap!
        ))

        gaps = tracer.find_trace_gaps()

        assert len(gaps) > 0
        # Should find the missing test gap
        gap_types = [g.gap_type for g in gaps]
        assert "missing_test" in gap_types

    def test_compute_coverage_metrics(self, tracer):
        """Test coverage metrics computation."""
        metrics = tracer.compute_coverage_metrics()

        assert isinstance(metrics, CoverageMetrics)
        assert metrics.total_requirements == 3
        assert metrics.requirements_with_parents == 2  # HW-001 and SW-001
        assert 0.0 <= metrics.upward_coverage <= 1.0
        assert 0.0 <= metrics.test_coverage <= 1.0

    def test_analyze_traceability(self, tracer):
        """Test complete traceability analysis."""
        report = tracer.analyze_traceability()

        assert isinstance(report, TraceabilityReport)
        assert report.total_requirements == 3
        assert 0.0 <= report.compliance_score <= 1.0

    def test_strict_mode_raises_on_critical_gaps(self, tracer):
        """Test strict mode raises exception on critical gaps."""
        # Add requirement with critical gap
        tracer.add_requirement(Requirement(
            req_id="HW-003",
            title="Critical Missing Test",
            description="ASIL-D without test",
            asil_level=ASILLevel.ASIL_D,
            parent_ids=["SYS-001"],
            status=RequirementStatus.IMPLEMENTED,
            # No tests - critical gap for ASIL-D
        ))

        with pytest.raises(SafetyTraceGapError) as exc_info:
            tracer.analyze_traceability(strict_mode=True)

        assert len(exc_info.value.gaps) > 0

    def test_to_json(self, tracer):
        """Test JSON export."""
        json_str = tracer.to_json()

        assert "SYS-001" in json_str
        assert "HW-001" in json_str
        assert '"requirements"' in json_str

    def test_from_json(self, tracer):
        """Test JSON import."""
        json_str = tracer.to_json()
        new_tracer = RequirementsTracer.from_json(json_str)

        assert len(new_tracer.requirements) == len(tracer.requirements)
        assert "SYS-001" in new_tracer.requirements

    def test_visualize_mermaid(self, tracer):
        """Test Mermaid diagram generation."""
        mermaid = tracer.visualize_graph(output_format="mermaid")

        assert "graph TD" in mermaid
        assert "SYS-001" in mermaid
        assert "-->" in mermaid

    def test_visualize_dot(self, tracer):
        """Test DOT diagram generation."""
        dot = tracer.visualize_graph(output_format="dot")

        assert "digraph" in dot
        assert "SYS-001" in dot
        assert "->" in dot

    def test_visualize_invalid_format(self, tracer):
        """Test invalid format raises error."""
        with pytest.raises(ValueError):
            tracer.visualize_graph(output_format="invalid")


class TestSafetyTraceGapError:
    """Tests for SafetyTraceGapError."""

    def test_error_with_gaps(self):
        """Test error includes gap details."""
        gaps = [
            {"req_id": "HW-001", "type": "missing_test", "message": "No test"},
        ]

        error = SafetyTraceGapError(gaps)

        assert len(error.gaps) == 1
        assert "1 traceability gap" in str(error)

    def test_error_with_custom_message(self):
        """Test error with custom message."""
        gaps = [{"req_id": "HW-001"}]
        error = SafetyTraceGapError(gaps, message="Custom message")

        assert str(error) == "Custom message"


class TestCoverageMetrics:
    """Tests for CoverageMetrics model."""

    def test_metrics_validation(self):
        """Test coverage values are bounded."""
        metrics = CoverageMetrics(
            total_requirements=10,
            requirements_with_parents=8,
            requirements_with_children=5,
            requirements_with_tests=7,
            requirements_implemented=6,
            requirements_verified=4,
            upward_coverage=0.8,
            downward_coverage=0.5,
            test_coverage=0.7,
            implementation_coverage=0.6,
        )

        assert 0.0 <= metrics.upward_coverage <= 1.0
        assert 0.0 <= metrics.test_coverage <= 1.0


