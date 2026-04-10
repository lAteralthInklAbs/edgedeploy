# ADR-004: ISO 26262 Requirements Traceability

## Status
Accepted

## Context
ISO 26262 (automotive functional safety) requires bidirectional traceability between requirements, design, implementation, and verification. Safety-critical ML systems must demonstrate complete traceability.

## Decision
Implement a requirements traceability engine with:

### Core Features
1. **Bidirectional Graph**: Parent→Child and Child→Parent traversal
2. **ASIL Levels**: Support QM, ASIL-A through ASIL-D
3. **BFS Traversal**: Impact analysis via breadth-first search
4. **Gap Detection**: Automated identification of traceability gaps

### Requirement Attributes
- Unique ID (pattern: XX-NNN, e.g., SYS-001, SW-012)
- ASIL level
- Status (draft → approved → implemented → verified → released)
- Parent/child links
- Test case references
- Source file references

### Compliance Checking
- Upward traceability (each requirement has parent)
- Downward traceability (system reqs decompose to SW/HW)
- Test coverage (especially for high-ASIL requirements)
- ASIL decomposition rules per ISO 26262

## Consequences

### Positive
- Automated compliance reporting
- Impact analysis for change management
- Clear audit trail for assessors
- Early detection of coverage gaps

### Negative
- Overhead in maintaining requirement metadata
- Requires discipline in keeping links updated
- May slow down rapid iteration phases

### Risks Mitigated
- Missing test coverage for safety requirements
- Orphaned requirements with no verification
- Invalid ASIL decomposition
- Audit failures during safety assessment

## ASIL Decomposition Rules
Per ISO 26262-9:

| Parent ASIL | Valid Decompositions |
|-------------|---------------------|
| ASIL D | D+QM, C+A, B+B |
| ASIL C | C+QM, B+A |
| ASIL B | B+QM, A+A |
| ASIL A | A+QM |

## Coverage Targets

| Metric | Target |
|--------|--------|
| Test coverage (overall) | ≥80% |
| ASIL-D coverage | 100% |
| Upward traceability | ≥70% |

## Implementation Notes

### Gap Severity
- **Critical**: ASIL-C/D requirement without test
- **Major**: ASIL-A/B requirement without test
- **Minor**: QM requirement without verification

### Strict Mode
When enabled, raises `SafetyTraceGapError` on any critical gaps, preventing deployment.

## Alternatives Considered

1. **Jama Connect**: Commercial tool, expensive licensing
2. **DOORS**: Legacy IBM tool, poor integration
3. **Spreadsheet-based**: Error-prone, no automation

## Related
- ISO 26262 Part 8: Supporting Processes
- ISO 26262 Part 6: Product Development at Software Level
- ADR-003: Canary Deployment


