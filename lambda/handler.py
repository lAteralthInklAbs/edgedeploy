"""
AWS Lambda handler for drift detection.

Triggered by CloudWatch Events on a schedule to monitor model drift.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import boto3
import numpy as np

# Initialize AWS clients
s3_client = boto3.client("s3")
sns_client = boto3.client("sns")

# Configuration from environment
S3_BUCKET = os.environ.get("S3_BUCKET", "edgedeploy-drift-data")
PSI_THRESHOLD = float(os.environ.get("PSI_THRESHOLD", "0.2"))
MMD_THRESHOLD = float(os.environ.get("MMD_THRESHOLD", "0.1"))
ALERT_SNS_TOPIC = os.environ.get("ALERT_SNS_TOPIC", "")


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index."""
    epsilon = 1e-6

    # Create bins based on expected distribution
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / len(expected) + epsilon
    actual_pct = actual_counts / len(actual) + epsilon

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def load_reference_data(key: str) -> np.ndarray:
    """Load reference distribution from S3."""
    response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    data = json.loads(response["Body"].read().decode("utf-8"))
    return np.array(data["values"])


def load_actual_data(key: str) -> np.ndarray:
    """Load actual distribution from S3."""
    response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    data = json.loads(response["Body"].read().decode("utf-8"))
    return np.array(data["values"])


def send_alert(drift_result: dict[str, Any]) -> None:
    """Send drift alert via SNS."""
    if not ALERT_SNS_TOPIC:
        print("No SNS topic configured, skipping alert")
        return

    message = {
        "alert_type": "DRIFT_DETECTED",
        "timestamp": datetime.utcnow().isoformat(),
        "psi_score": drift_result["psi_score"],
        "threshold": PSI_THRESHOLD,
        "level": drift_result["level"],
        "message": drift_result["message"],
    }

    sns_client.publish(
        TopicArn=ALERT_SNS_TOPIC,
        Subject="EdgeDeploy Drift Alert",
        Message=json.dumps(message, indent=2),
    )


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Lambda handler for drift detection.

    Args:
        event: CloudWatch event or manual trigger
        context: Lambda context

    Returns:
        Drift detection result
    """
    print(f"Drift detection triggered at {datetime.utcnow().isoformat()}")

    try:
        # Load reference and actual data
        reference_key = event.get("reference_key", "reference/latest.json")
        actual_key = event.get("actual_key", "actual/latest.json")

        reference_data = load_reference_data(reference_key)
        actual_data = load_actual_data(actual_key)

        print(f"Reference samples: {len(reference_data)}")
        print(f"Actual samples: {len(actual_data)}")

        # Compute PSI
        psi_score = compute_psi(reference_data, actual_data)
        print(f"PSI Score: {psi_score:.4f}")

        # Determine drift level
        if psi_score < 0.1:
            level = "none"
            drift_detected = False
            message = "No significant drift detected"
        elif psi_score < PSI_THRESHOLD:
            level = "low"
            drift_detected = False
            message = "Minor distribution shift, monitoring"
        elif psi_score < 0.3:
            level = "medium"
            drift_detected = True
            message = "Moderate drift detected, consider retraining"
        else:
            level = "high"
            drift_detected = True
            message = "Significant drift detected, retraining recommended"

        result = {
            "drift_detected": drift_detected,
            "level": level,
            "psi_score": psi_score,
            "psi_threshold": PSI_THRESHOLD,
            "reference_samples": len(reference_data),
            "actual_samples": len(actual_data),
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Send alert if drift detected
        if drift_detected:
            send_alert(result)

        # Store result
        result_key = f"results/{datetime.utcnow().strftime('%Y/%m/%d/%H%M%S')}.json"
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=result_key,
            Body=json.dumps(result, indent=2),
            ContentType="application/json",
        )

        print(f"Result stored at s3://{S3_BUCKET}/{result_key}")

        return {
            "statusCode": 200,
            "body": json.dumps(result),
        }

    except Exception as e:
        error_result = {
            "statusCode": 500,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }
        print(f"Error: {e}")
        return error_result


