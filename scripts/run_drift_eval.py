#!/usr/bin/env python3
"""
Drift detection evaluation script.

Demonstrates PSI, MMD, and ADWIN ensemble drift detection.

Usage:
    python scripts/run_drift_eval.py [--scenarios all] [--output results.json]

Metrics produced:
    - PSI scores for each scenario
    - MMD scores for each scenario
    - ADWIN detection results
    - Ensemble weighted scores
    - Detection accuracy vs ground truth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.drift.drift_ensemble import (
    DriftEnsemble,
    DriftLevel,
    ADWINDetector,
)
from fixtures.drift_scenarios import (
    get_all_scenarios,
    DriftScenario,
    create_streaming_drift_scenario,
)


SEED = 42


def evaluate_scenario(
    scenario: DriftScenario,
    ensemble: DriftEnsemble,
) -> dict[str, Any]:
    """Evaluate a single drift scenario."""
    result = ensemble.detect(scenario.actual_data)

    # Determine if detection matches expected
    expected_drift = scenario.expected_drift
    detected_drift = result.level != DriftLevel.NONE

    correct = (expected_drift == detected_drift)

    # Level accuracy (more nuanced)
    level_order = {"none": 0, "warning": 1, "critical": 2}
    expected_level = level_order.get(scenario.expected_level, 0)
    detected_level = level_order.get(result.level.value, 0)
    level_diff = abs(expected_level - detected_level)

    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "expected_drift": expected_drift,
        "detected_drift": detected_drift,
        "expected_level": scenario.expected_level,
        "detected_level": result.level.value,
        "correct_detection": correct,
        "level_accuracy": 1.0 - (level_diff / 2),
        "psi_score": result.psi_score,
        "mmd_score": result.mmd_score,
        "ensemble_score": result.ensemble_score,
        "explanation": result.explanation,
    }


def print_scenario_result(result: dict[str, Any]) -> None:
    """Print formatted scenario result."""
    status = "✓" if result["correct_detection"] else "✗"

    print(f"\n{status} {result['scenario'].upper()}")
    print(f"  Description: {result['description']}")
    print(f"  Expected: drift={result['expected_drift']}, level={result['expected_level']}")
    print(f"  Detected: drift={result['detected_drift']}, level={result['detected_level']}")
    print(f"  Scores: PSI={result['psi_score']:.4f}, MMD={result['mmd_score']:.4f}, Ensemble={result['ensemble_score']:.4f}")


def evaluate_streaming_detection() -> dict[str, Any]:
    """Evaluate ADWIN on streaming data with known drift point."""
    print("\n" + "=" * 60)
    print("STREAMING DRIFT DETECTION (ADWIN)")
    print("=" * 60)

    scenario = create_streaming_drift_scenario(
        n_samples=2000,
        drift_point=1000,
        seed=SEED,
    )

    stream = scenario.metadata["full_stream"]
    drift_point = scenario.metadata["drift_point"]

    detector = ADWINDetector(delta=0.002, max_window_size=500)

    first_detection = None
    detection_count = 0

    for i, value in enumerate(stream.flatten()):
        detected = detector.add(float(value))
        if detected:
            detection_count += 1
            if first_detection is None:
                first_detection = i

    print(f"\n  True drift point: {drift_point}")
    print(f"  First detection:  {first_detection}")
    print(f"  Total detections: {detection_count}")

    if first_detection is not None:
        delay = first_detection - drift_point
        print(f"  Detection delay:  {delay} samples")

        return {
            "true_drift_point": drift_point,
            "first_detection": first_detection,
            "detection_delay": delay,
            "total_detections": detection_count,
            "accurate": abs(delay) < 200,  # Within 200 samples
        }
    else:
        print("  WARNING: Drift not detected!")
        return {
            "true_drift_point": drift_point,
            "first_detection": None,
            "detection_delay": None,
            "total_detections": 0,
            "accurate": False,
        }


def main() -> None:
    """Run drift detection evaluation."""
    parser = argparse.ArgumentParser(description="Drift Detection Evaluation")
    parser.add_argument(
        "--scenarios",
        type=str,
        default="all",
        help="Scenarios to evaluate (all, or comma-separated names)",
    )
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    print("=" * 60)
    print("EdgeDeploy - Drift Detection Evaluation")
    print("=" * 60)

    np.random.seed(SEED)

    # Get scenarios
    all_scenarios = get_all_scenarios()

    if args.scenarios != "all":
        scenario_names = args.scenarios.split(",")
        scenarios = [s for s in all_scenarios if s.name in scenario_names]
    else:
        scenarios = all_scenarios

    print(f"\nEvaluating {len(scenarios)} drift scenarios...")

    # Evaluate each scenario
    results = []
    for scenario in scenarios:
        # Create fresh ensemble with reference data
        ensemble = DriftEnsemble(persistence_required=1)
        ensemble.fit(scenario.reference_data)

        result = evaluate_scenario(scenario, ensemble)
        results.append(result)
        print_scenario_result(result)

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    correct_count = sum(1 for r in results if r["correct_detection"])
    total_count = len(results)
    accuracy = correct_count / total_count

    avg_level_accuracy = sum(r["level_accuracy"] for r in results) / total_count

    print(f"\n  Detection Accuracy: {accuracy:.1%} ({correct_count}/{total_count})")
    print(f"  Level Accuracy:     {avg_level_accuracy:.1%}")

    # Per-drift-type breakdown
    drift_results = [r for r in results if r["expected_drift"]]
    no_drift_results = [r for r in results if not r["expected_drift"]]

    if drift_results:
        drift_accuracy = sum(1 for r in drift_results if r["correct_detection"]) / len(drift_results)
        print(f"  True Drift Recall:  {drift_accuracy:.1%}")

    if no_drift_results:
        no_drift_accuracy = sum(1 for r in no_drift_results if r["correct_detection"]) / len(no_drift_results)
        print(f"  No Drift Precision: {no_drift_accuracy:.1%}")

    # Streaming evaluation
    streaming_result = evaluate_streaming_detection()

    # Save results
    if args.output:
        output_data = {
            "overall_accuracy": accuracy,
            "level_accuracy": avg_level_accuracy,
            "scenario_results": results,
            "streaming_result": streaming_result,
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 60)
    print("[DONE] Drift detection evaluation complete.")


if __name__ == "__main__":
    main()


