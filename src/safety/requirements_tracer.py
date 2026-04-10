"""
ISO 26262 Requirements Traceability Engine.

★DEEP MODULE - Contains substantial custom logic for:
- Bidirectional requirements graph (parent→child, child→parent)
- BFS traversal for impact analysis
- Completeness checking against ASIL levels
- Trace gap detection with detailed reporting
- Coverage metrics computation
- Requirements status lifecycle management
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import json
import hashlib
from datetime import datetime

from pydantic import BaseModel, Field


class RequirementStatus(str, Enum):
    """Lifecycle status of a requirement."""

    DRAFT = "draft"
    APPROVED = "approved"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"
    RELEASED = "released"
    DEPRECATED = "deprecated"


class ASILLevel(str, Enum):
    """Automotive Safety Integrity Level per ISO 26262."""

    QM = "QM"  # Quality Management (no safety requirements)
    ASIL_A = "ASIL_A"
    ASIL_B = "ASIL_B"
    ASIL_C = "ASIL_C"
    ASIL_D = "ASIL_D"  # Most stringent


# ASIL decomposition rules per ISO 26262
ASIL_DECOMPOSITION_RULES: dict[ASILLevel, list[tuple[ASILLevel, ASILLevel]]] = {
    ASILLevel.ASIL_D: [
        (ASILLevel.ASIL_D, ASILLevel.QM),
        (ASILLevel.ASIL_C, ASILLevel.ASIL_A),
        (ASILLevel.ASIL_B, ASILLevel.ASIL_B),
    ],
    ASILLevel.ASIL_C: [
        (ASILLevel.ASIL_C, ASILLevel.QM),
        (ASILLevel.ASIL_B, ASILLevel.ASIL_A),
    ],
    ASILLevel.ASIL_B: [
        (ASILLevel.ASIL_B, ASILLevel.QM),
        (ASILLevel.ASIL_A, ASILLevel.ASIL_A),
    ],
    ASILLevel.ASIL_A: [
        (ASILLevel.ASIL_A, ASILLevel.QM),
    ],
    ASILLevel.QM: [],
}


class SafetyTraceGapError(Exception):
    """Raised when safety-critical requirements have traceability gaps."""

    def __init__(self, gaps: list[dict[str, Any]], message: str | None = None) -> None:
        self.gaps = gaps
        if message is None:
            message = f"Found {len(gaps)} traceability gap(s) in safety requirements"
        super().__init__(message)


class Requirement(BaseModel):
    """Single requirement with traceability metadata."""

    req_id: str = Field(..., pattern=r"^[A-Z]{2,4}-\d{3,6}$")
    title: str = Field(..., min_length=5, max_length=200)
    description: str
    asil_level: ASILLevel
    status: RequirementStatus = RequirementStatus.DRAFT
    parent_ids: list[str] = Field(default_factory=list)
    child_ids: list[str] = Field(default_factory=list)
    test_case_ids: list[str] = Field(default_factory=list)
    source_file_refs: list[str] = Field(default_factory=list)
    verification_method: str = Field(default="test")  # test, review, analysis, demo
    rationale: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1

    def content_hash(self) -> str:
        """Compute hash of requirement content for change detection."""
        content = f"{self.title}|{self.description}|{self.asil_level.value}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]


class TraceLink(BaseModel):
    """Bidirectional link between requirements or artifacts."""

    source_id: str
    target_id: str
    link_type: str = Field(default="derives")  # derives, refines, verifies, implements
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    rationale: str = ""


class TraceGap(BaseModel):
    """Detected gap in traceability."""

    req_id: str
    gap_type: str  # missing_parent, missing_child, missing_test, missing_impl
    severity: str  # critical, major, minor
    asil_level: ASILLevel
    message: str
    suggested_action: str


class CoverageMetrics(BaseModel):
    """Traceability coverage metrics."""

    total_requirements: int
    requirements_with_parents: int
    requirements_with_children: int
    requirements_with_tests: int
    requirements_implemented: int
    requirements_verified: int

    # Per-ASIL coverage
    asil_d_coverage: float = 0.0
    asil_c_coverage: float = 0.0
    asil_b_coverage: float = 0.0
    asil_a_coverage: float = 0.0
    qm_coverage: float = 0.0

    # Overall metrics
    upward_coverage: float = Field(ge=0.0, le=1.0)
    downward_coverage: float = Field(ge=0.0, le=1.0)
    test_coverage: float = Field(ge=0.0, le=1.0)
    implementation_coverage: float = Field(ge=0.0, le=1.0)


class TraceabilityReport(BaseModel):
    """Complete traceability analysis report."""

    generated_at: datetime = Field(default_factory=datetime.utcnow)
    total_requirements: int
    gaps: list[TraceGap]
    metrics: CoverageMetrics
    critical_gaps_count: int
    is_compliant: bool
    compliance_score: float = Field(ge=0.0, le=1.0)
    recommendations: list[str]


def _compute_asil_weight(level: ASILLevel) -> int:
    """Get numeric weight for ASIL level (higher = more critical)."""
    weights = {
        ASILLevel.QM: 0,
        ASILLevel.ASIL_A: 1,
        ASILLevel.ASIL_B: 2,
        ASILLevel.ASIL_C: 3,
        ASILLevel.ASIL_D: 4,
    }
    return weights.get(level, 0)


def _gap_severity_from_asil(level: ASILLevel, gap_type: str) -> str:
    """Determine gap severity based on ASIL level and gap type."""
    weight = _compute_asil_weight(level)

    if gap_type in ("missing_parent", "missing_test"):
        if weight >= 3:  # ASIL_C or ASIL_D
            return "critical"
        elif weight >= 1:  # ASIL_A or ASIL_B
            return "major"
    elif gap_type == "missing_impl":
        if weight >= 2:  # ASIL_B, C, D
            return "critical"
        elif weight >= 1:
            return "major"

    return "minor"


@dataclass
class RequirementsTracer:
    """
    ISO 26262 requirements traceability engine.

    Provides:
    - Bidirectional requirements graph management
    - BFS-based impact analysis
    - Completeness checking per ASIL level
    - Coverage metrics computation
    - Gap detection and reporting

    Example:
        tracer = RequirementsTracer()
        tracer.add_requirement(Requirement(
            req_id="SYS-001",
            title="System Safety Function",
            description="The system shall...",
            asil_level=ASILLevel.ASIL_D,
        ))
        report = tracer.analyze_traceability()
    """

    requirements: dict[str, Requirement] = field(default_factory=dict)
    trace_links: list[TraceLink] = field(default_factory=list)

    # Graph structures
    _parent_graph: dict[str, set[str]] = field(default_factory=dict)
    _child_graph: dict[str, set[str]] = field(default_factory=dict)
    _test_graph: dict[str, set[str]] = field(default_factory=dict)

    def add_requirement(self, req: Requirement) -> None:
        """Add a requirement to the tracer."""
        self.requirements[req.req_id] = req

        # Initialize graph entries
        if req.req_id not in self._parent_graph:
            self._parent_graph[req.req_id] = set()
        if req.req_id not in self._child_graph:
            self._child_graph[req.req_id] = set()
        if req.req_id not in self._test_graph:
            self._test_graph[req.req_id] = set()

        # Add parent links
        for parent_id in req.parent_ids:
            self._parent_graph[req.req_id].add(parent_id)
            if parent_id not in self._child_graph:
                self._child_graph[parent_id] = set()
            self._child_graph[parent_id].add(req.req_id)

        # Add child links
        for child_id in req.child_ids:
            self._child_graph[req.req_id].add(child_id)
            if child_id not in self._parent_graph:
                self._parent_graph[child_id] = set()
            self._parent_graph[child_id].add(req.req_id)

        # Add test links
        for test_id in req.test_case_ids:
            self._test_graph[req.req_id].add(test_id)

    def add_trace_link(self, link: TraceLink) -> None:
        """Add a trace link between artifacts."""
        self.trace_links.append(link)

        if link.link_type == "derives":
            if link.target_id not in self._parent_graph:
                self._parent_graph[link.target_id] = set()
            self._parent_graph[link.target_id].add(link.source_id)

            if link.source_id not in self._child_graph:
                self._child_graph[link.source_id] = set()
            self._child_graph[link.source_id].add(link.target_id)

        elif link.link_type == "verifies":
            if link.source_id not in self._test_graph:
                self._test_graph[link.source_id] = set()
            self._test_graph[link.source_id].add(link.target_id)

    def get_ancestors(self, req_id: str) -> list[str]:
        """
        Get all ancestor requirements using BFS traversal.

        Traverses upward through parent links to find all ancestors.
        """
        if req_id not in self._parent_graph:
            return []

        ancestors: list[str] = []
        visited: set[str] = set()
        queue: deque[str] = deque()

        # Start with immediate parents
        for parent_id in self._parent_graph.get(req_id, set()):
            if parent_id not in visited:
                queue.append(parent_id)
                visited.add(parent_id)

        while queue:
            current = queue.popleft()
            ancestors.append(current)

            # Add parents of current
            for parent_id in self._parent_graph.get(current, set()):
                if parent_id not in visited:
                    queue.append(parent_id)
                    visited.add(parent_id)

        return ancestors

    def get_descendants(self, req_id: str) -> list[str]:
        """
        Get all descendant requirements using BFS traversal.

        Traverses downward through child links to find all descendants.
        """
        if req_id not in self._child_graph:
            return []

        descendants: list[str] = []
        visited: set[str] = set()
        queue: deque[str] = deque()

        # Start with immediate children
        for child_id in self._child_graph.get(req_id, set()):
            if child_id not in visited:
                queue.append(child_id)
                visited.add(child_id)

        while queue:
            current = queue.popleft()
            descendants.append(current)

            # Add children of current
            for child_id in self._child_graph.get(current, set()):
                if child_id not in visited:
                    queue.append(child_id)
                    visited.add(child_id)

        return descendants

    def impact_analysis(self, req_id: str) -> dict[str, Any]:
        """
        Perform impact analysis for a requirement change.

        Uses BFS to find all affected requirements (ancestors and descendants).

        Args:
            req_id: Requirement ID to analyze

        Returns:
            Dict with affected requirements, test cases, and ASIL impact
        """
        if req_id not in self.requirements:
            return {"error": f"Requirement {req_id} not found"}

        req = self.requirements[req_id]
        ancestors = self.get_ancestors(req_id)
        descendants = self.get_descendants(req_id)

        # Collect affected test cases
        affected_tests: set[str] = set()
        affected_tests.update(self._test_graph.get(req_id, set()))

        for desc_id in descendants:
            affected_tests.update(self._test_graph.get(desc_id, set()))

        # Determine highest ASIL level affected
        max_asil_weight = _compute_asil_weight(req.asil_level)
        for desc_id in descendants:
            if desc_id in self.requirements:
                weight = _compute_asil_weight(self.requirements[desc_id].asil_level)
                max_asil_weight = max(max_asil_weight, weight)

        return {
            "requirement_id": req_id,
            "ancestors": ancestors,
            "descendants": descendants,
            "total_affected": len(ancestors) + len(descendants) + 1,
            "affected_tests": list(affected_tests),
            "highest_asil_affected": max_asil_weight,
            "propagation_depth_up": len(ancestors),
            "propagation_depth_down": len(descendants),
        }

    def check_asil_decomposition(self, parent_id: str, child_ids: list[str]) -> dict[str, Any]:
        """
        Verify ASIL decomposition is valid per ISO 26262.

        Parent ASIL must be achievable from children's combined ASIL levels.

        Args:
            parent_id: Parent requirement ID
            child_ids: List of child requirement IDs

        Returns:
            Dict with validation result and details
        """
        if parent_id not in self.requirements:
            return {"valid": False, "error": f"Parent {parent_id} not found"}

        parent_asil = self.requirements[parent_id].asil_level
        child_asils = []

        for child_id in child_ids:
            if child_id in self.requirements:
                child_asils.append(self.requirements[child_id].asil_level)

        if not child_asils:
            return {"valid": False, "error": "No valid children found"}

        # Check if decomposition is valid
        allowed_decomps = ASIL_DECOMPOSITION_RULES.get(parent_asil, [])

        # For single child, it must match parent ASIL
        if len(child_asils) == 1:
            is_valid = child_asils[0] == parent_asil
            return {
                "valid": is_valid,
                "parent_asil": parent_asil.value,
                "child_asils": [a.value for a in child_asils],
                "message": "Single child must have same ASIL as parent" if not is_valid else "Valid",
            }

        # For two children, check decomposition rules
        if len(child_asils) == 2:
            pair = (child_asils[0], child_asils[1])
            pair_reverse = (child_asils[1], child_asils[0])
            is_valid = pair in allowed_decomps or pair_reverse in allowed_decomps

            return {
                "valid": is_valid,
                "parent_asil": parent_asil.value,
                "child_asils": [a.value for a in child_asils],
                "allowed_decompositions": [(a.value, b.value) for a, b in allowed_decomps],
                "message": "Valid decomposition" if is_valid else "Invalid ASIL decomposition",
            }

        # For more than 2 children, check if highest child ASIL >= parent ASIL
        max_child_weight = max(_compute_asil_weight(a) for a in child_asils)
        parent_weight = _compute_asil_weight(parent_asil)

        return {
            "valid": max_child_weight >= parent_weight,
            "parent_asil": parent_asil.value,
            "child_asils": [a.value for a in child_asils],
            "message": "At least one child must have ASIL >= parent",
        }

    def find_trace_gaps(self) -> list[TraceGap]:
        """
        Find all traceability gaps in the requirements graph.

        Checks for:
        - Missing parent requirements (upward traceability)
        - Missing child requirements (downward traceability)
        - Missing test cases
        - Missing implementation references

        Returns:
            List of detected trace gaps
        """
        gaps: list[TraceGap] = []

        for req_id, req in self.requirements.items():
            # Check for missing parents (except top-level)
            if not self._parent_graph.get(req_id) and req.asil_level != ASILLevel.QM:
                # Determine if this should have a parent
                if not req_id.startswith("SYS-"):  # System-level can be top
                    gap_type = "missing_parent"
                    severity = _gap_severity_from_asil(req.asil_level, gap_type)
                    gaps.append(TraceGap(
                        req_id=req_id,
                        gap_type=gap_type,
                        severity=severity,
                        asil_level=req.asil_level,
                        message=f"Requirement {req_id} has no parent (upward trace missing)",
                        suggested_action="Add parent requirement link or mark as top-level",
                    ))

            # Check for missing tests on implemented requirements
            if req.status in (RequirementStatus.IMPLEMENTED, RequirementStatus.VERIFIED):
                if not self._test_graph.get(req_id) and not req.test_case_ids:
                    gap_type = "missing_test"
                    severity = _gap_severity_from_asil(req.asil_level, gap_type)
                    gaps.append(TraceGap(
                        req_id=req_id,
                        gap_type=gap_type,
                        severity=severity,
                        asil_level=req.asil_level,
                        message=f"Requirement {req_id} is implemented but has no test cases",
                        suggested_action="Add test case references",
                    ))

            # Check for missing implementation on approved requirements
            if req.status == RequirementStatus.APPROVED:
                if not req.source_file_refs:
                    gap_type = "missing_impl"
                    severity = _gap_severity_from_asil(req.asil_level, gap_type)
                    gaps.append(TraceGap(
                        req_id=req_id,
                        gap_type=gap_type,
                        severity=severity,
                        asil_level=req.asil_level,
                        message=f"Requirement {req_id} is approved but has no implementation reference",
                        suggested_action="Add source file references after implementation",
                    ))

            # Check leaf nodes in high-ASIL requirements
            is_leaf = not self._child_graph.get(req_id)
            if is_leaf and _compute_asil_weight(req.asil_level) >= 3:
                # Leaf ASIL-C/D requirements should be directly testable
                if not self._test_graph.get(req_id) and req.verification_method == "test":
                    gaps.append(TraceGap(
                        req_id=req_id,
                        gap_type="missing_test",
                        severity="critical",
                        asil_level=req.asil_level,
                        message=f"Leaf requirement {req_id} (ASIL-C/D) requires direct test verification",
                        suggested_action="Add test cases for leaf-level safety requirement",
                    ))

        # Sort by severity (critical first)
        severity_order = {"critical": 0, "major": 1, "minor": 2}
        gaps.sort(key=lambda g: severity_order.get(g.severity, 3))

        return gaps

    def compute_coverage_metrics(self) -> CoverageMetrics:
        """
        Compute traceability coverage metrics.

        Returns:
            CoverageMetrics with detailed coverage information
        """
        total = len(self.requirements)
        if total == 0:
            return CoverageMetrics(
                total_requirements=0,
                requirements_with_parents=0,
                requirements_with_children=0,
                requirements_with_tests=0,
                requirements_implemented=0,
                requirements_verified=0,
                upward_coverage=0.0,
                downward_coverage=0.0,
                test_coverage=0.0,
                implementation_coverage=0.0,
            )

        with_parents = sum(
            1 for req_id in self.requirements
            if self._parent_graph.get(req_id)
        )
        with_children = sum(
            1 for req_id in self.requirements
            if self._child_graph.get(req_id)
        )
        with_tests = sum(
            1 for req_id in self.requirements
            if self._test_graph.get(req_id) or self.requirements[req_id].test_case_ids
        )
        implemented = sum(
            1 for req in self.requirements.values()
            if req.status in (
                RequirementStatus.IMPLEMENTED,
                RequirementStatus.VERIFIED,
                RequirementStatus.RELEASED,
            )
        )
        verified = sum(
            1 for req in self.requirements.values()
            if req.status in (RequirementStatus.VERIFIED, RequirementStatus.RELEASED)
        )

        # Per-ASIL coverage (percentage with tests)
        asil_counts: dict[ASILLevel, tuple[int, int]] = {
            level: (0, 0) for level in ASILLevel
        }
        for req in self.requirements.values():
            total_count, tested_count = asil_counts[req.asil_level]
            has_test = (
                self._test_graph.get(req.req_id) or
                bool(req.test_case_ids)
            )
            asil_counts[req.asil_level] = (
                total_count + 1,
                tested_count + (1 if has_test else 0),
            )

        def _safe_ratio(numerator: int, denominator: int) -> float:
            return numerator / denominator if denominator > 0 else 0.0

        return CoverageMetrics(
            total_requirements=total,
            requirements_with_parents=with_parents,
            requirements_with_children=with_children,
            requirements_with_tests=with_tests,
            requirements_implemented=implemented,
            requirements_verified=verified,
            asil_d_coverage=_safe_ratio(*asil_counts[ASILLevel.ASIL_D]),
            asil_c_coverage=_safe_ratio(*asil_counts[ASILLevel.ASIL_C]),
            asil_b_coverage=_safe_ratio(*asil_counts[ASILLevel.ASIL_B]),
            asil_a_coverage=_safe_ratio(*asil_counts[ASILLevel.ASIL_A]),
            qm_coverage=_safe_ratio(*asil_counts[ASILLevel.QM]),
            upward_coverage=with_parents / total,
            downward_coverage=with_children / total,
            test_coverage=with_tests / total,
            implementation_coverage=implemented / total,
        )

    def analyze_traceability(
        self,
        strict_mode: bool = False,
    ) -> TraceabilityReport:
        """
        Perform complete traceability analysis.

        Args:
            strict_mode: If True, raises SafetyTraceGapError on critical gaps

        Returns:
            Complete TraceabilityReport

        Raises:
            SafetyTraceGapError: In strict mode if critical gaps found
        """
        gaps = self.find_trace_gaps()
        metrics = self.compute_coverage_metrics()

        critical_count = sum(1 for g in gaps if g.severity == "critical")
        major_count = sum(1 for g in gaps if g.severity == "major")

        # Compliance scoring
        # Deduct points for gaps: critical=-20, major=-10, minor=-5
        base_score = 100.0
        score = base_score - (critical_count * 20) - (major_count * 10)
        score -= sum(5 for g in gaps if g.severity == "minor")
        score = max(0.0, min(100.0, score)) / 100.0

        # Compliance requires no critical gaps and >80% coverage
        is_compliant = (
            critical_count == 0 and
            metrics.test_coverage >= 0.8 and
            metrics.upward_coverage >= 0.7
        )

        # Generate recommendations
        recommendations: list[str] = []

        if critical_count > 0:
            recommendations.append(
                f"URGENT: Address {critical_count} critical traceability gap(s)"
            )
        if metrics.test_coverage < 0.8:
            recommendations.append(
                f"Improve test coverage from {metrics.test_coverage:.0%} to >80%"
            )
        if metrics.asil_d_coverage < 1.0:
            recommendations.append(
                "Ensure 100% test coverage for ASIL-D requirements"
            )
        if metrics.upward_coverage < 0.7:
            recommendations.append(
                "Improve upward traceability - link requirements to parents"
            )

        report = TraceabilityReport(
            total_requirements=metrics.total_requirements,
            gaps=gaps,
            metrics=metrics,
            critical_gaps_count=critical_count,
            is_compliant=is_compliant,
            compliance_score=score,
            recommendations=recommendations,
        )

        if strict_mode and critical_count > 0:
            gap_details = [
                {"req_id": g.req_id, "type": g.gap_type, "message": g.message}
                for g in gaps
                if g.severity == "critical"
            ]
            raise SafetyTraceGapError(gap_details)

        return report

    def to_json(self) -> str:
        """Export requirements graph to JSON."""
        data = {
            "requirements": [
                req.model_dump(mode="json") for req in self.requirements.values()
            ],
            "trace_links": [
                link.model_dump(mode="json") for link in self.trace_links
            ],
        }
        return json.dumps(data, indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "RequirementsTracer":
        """Import requirements graph from JSON."""
        data = json.loads(json_str)
        tracer = cls()

        for req_data in data.get("requirements", []):
            req = Requirement(**req_data)
            tracer.add_requirement(req)

        for link_data in data.get("trace_links", []):
            link = TraceLink(**link_data)
            tracer.add_trace_link(link)

        return tracer

    def visualize_graph(self, output_format: str = "mermaid") -> str:
        """
        Generate visualization of requirements graph.

        Args:
            output_format: 'mermaid' or 'dot'

        Returns:
            Graph definition string
        """
        if output_format == "mermaid":
            lines = ["graph TD"]

            # Add nodes with ASIL styling
            for req_id, req in self.requirements.items():
                style = {
                    ASILLevel.ASIL_D: ":::asild",
                    ASILLevel.ASIL_C: ":::asilc",
                    ASILLevel.ASIL_B: ":::asilb",
                    ASILLevel.ASIL_A: ":::asila",
                    ASILLevel.QM: ":::qm",
                }.get(req.asil_level, "")

                safe_title = req.title[:30].replace('"', "'")
                lines.append(f'    {req_id}["{req_id}: {safe_title}"]{style}')

            # Add edges
            for req_id in self.requirements:
                for child_id in self._child_graph.get(req_id, set()):
                    if child_id in self.requirements:
                        lines.append(f"    {req_id} --> {child_id}")

            # Add styles
            lines.extend([
                "    classDef asild fill:#ff6b6b,stroke:#333",
                "    classDef asilc fill:#ffa94d,stroke:#333",
                "    classDef asilb fill:#ffe066,stroke:#333",
                "    classDef asila fill:#8ce99a,stroke:#333",
                "    classDef qm fill:#e9ecef,stroke:#333",
            ])

            return "\n".join(lines)

        elif output_format == "dot":
            lines = ["digraph requirements {", "    rankdir=TB;"]

            for req_id, req in self.requirements.items():
                color = {
                    ASILLevel.ASIL_D: "red",
                    ASILLevel.ASIL_C: "orange",
                    ASILLevel.ASIL_B: "yellow",
                    ASILLevel.ASIL_A: "green",
                    ASILLevel.QM: "gray",
                }.get(req.asil_level, "white")

                lines.append(f'    "{req_id}" [fillcolor={color}, style=filled];')

            for req_id in self.requirements:
                for child_id in self._child_graph.get(req_id, set()):
                    if child_id in self.requirements:
                        lines.append(f'    "{req_id}" -> "{child_id}";')

            lines.append("}")
            return "\n".join(lines)

        else:
            raise ValueError(f"Unknown output format: {output_format}")


