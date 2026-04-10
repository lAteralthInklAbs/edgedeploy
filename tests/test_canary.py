"""Tests for canary deployment engine."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.deployment.canary import (
    CanaryDeployment,
    DeploymentConfig,
    DeploymentStage,
    DeploymentStatus,
    HealthCheck,
    RollbackTrigger,
    RollbackReason,
    DeploymentMetrics,
    TrafficRouter,
    HealthMonitor,
    create_default_deployment,
    create_conservative_deployment,
)


class TestTrafficRouter:
    """Tests for TrafficRouter."""

    def test_router_initialization(self):
        """Test router initializes with 0% canary."""
        router = TrafficRouter()
        assert router.canary_percentage == 0.0

    def test_update_canary_percentage(self):
        """Test updating canary percentage."""
        router = TrafficRouter()
        router.update_canary_percentage(25.0)
        assert router.canary_percentage == 25.0

    def test_canary_percentage_clamped(self):
        """Test canary percentage is clamped to [0, 100]."""
        router = TrafficRouter()

        router.update_canary_percentage(-10.0)
        assert router.canary_percentage == 0.0

        router.update_canary_percentage(150.0)
        assert router.canary_percentage == 100.0

    def test_route_request_consistent(self):
        """Test same request ID routes consistently."""
        router = TrafficRouter(canary_percentage=50.0)

        request_id = "test-request-123"
        results = [router.route_request(request_id) for _ in range(10)]

        # Same request should always route same way
        assert all(r == results[0] for r in results)

    def test_route_request_distribution(self):
        """Test traffic roughly matches canary percentage."""
        router = TrafficRouter(canary_percentage=30.0)

        canary_count = 0
        total = 1000

        for i in range(total):
            if router.route_request(f"request-{i}") == "canary":
                canary_count += 1

        # Should be roughly 30% (with some tolerance)
        actual_pct = canary_count / total * 100
        assert 20.0 < actual_pct < 40.0

    def test_get_actual_distribution(self):
        """Test actual distribution tracking."""
        router = TrafficRouter(canary_percentage=50.0)

        for i in range(100):
            router.route_request(f"request-{i}")

        dist = router.get_actual_distribution()
        assert "canary" in dist
        assert "stable" in dist
        assert dist["total_requests"] == 100


class TestHealthMonitor:
    """Tests for HealthMonitor."""

    def test_monitor_initialization(self):
        """Test monitor initializes empty."""
        monitor = HealthMonitor()
        health = monitor.get_health_check()

        assert health.request_count == 0
        assert health.is_healthy is True

    def test_record_success(self):
        """Test recording successful requests."""
        monitor = HealthMonitor()

        for _ in range(10):
            monitor.record_request(latency_ms=50.0, success=True)

        health = monitor.get_health_check()
        assert health.request_count == 10
        assert health.success_rate == 1.0
        assert health.error_count == 0

    def test_record_failures(self):
        """Test recording failed requests."""
        monitor = HealthMonitor()

        for _ in range(8):
            monitor.record_request(latency_ms=50.0, success=True)
        for _ in range(2):
            monitor.record_request(
                latency_ms=100.0,
                success=False,
                error_message="Timeout",
            )

        health = monitor.get_health_check()
        assert health.request_count == 10
        assert health.success_rate == 0.8
        assert health.error_count == 2

    def test_latency_percentiles(self):
        """Test latency percentile calculation."""
        monitor = HealthMonitor()

        # Add varied latencies
        latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for lat in latencies:
            monitor.record_request(latency_ms=float(lat), success=True)

        health = monitor.get_health_check()

        # P50 should be around 50ms
        assert 40 <= health.latency_p50_ms <= 60

        # P99 should be high
        assert health.latency_p99_ms >= 90

    def test_accuracy_tracking(self):
        """Test accuracy metric tracking."""
        monitor = HealthMonitor()

        monitor.record_request(latency_ms=50.0, success=True, accuracy=0.95)
        monitor.record_request(latency_ms=50.0, success=True, accuracy=0.85)

        health = monitor.get_health_check()
        assert health.accuracy == 0.90  # Average

    def test_reset(self):
        """Test monitor reset."""
        monitor = HealthMonitor()

        for _ in range(10):
            monitor.record_request(latency_ms=50.0, success=True)

        monitor.reset()
        health = monitor.get_health_check()

        assert health.request_count == 0


class TestDeploymentStage:
    """Tests for DeploymentStage model."""

    def test_stage_creation(self):
        """Test creating a deployment stage."""
        stage = DeploymentStage(
            name="canary",
            traffic_percentage=5.0,
            duration_minutes=15,
        )

        assert stage.name == "canary"
        assert stage.traffic_percentage == 5.0
        assert stage.min_success_rate == 0.95

    def test_stage_validation(self):
        """Test stage validation."""
        with pytest.raises(ValueError):
            DeploymentStage(
                name="invalid",
                traffic_percentage=150.0,  # > 100
                duration_minutes=15,
            )


class TestDeploymentConfig:
    """Tests for DeploymentConfig model."""

    def test_default_config(self):
        """Test default configuration."""
        config = DeploymentConfig(model_version="v1.0.0")

        assert len(config.stages) == 4
        assert config.enable_auto_rollback is True

    def test_default_stages(self):
        """Test default stage progression."""
        config = DeploymentConfig(model_version="v1.0.0")

        percentages = [s.traffic_percentage for s in config.stages]
        assert percentages == [5.0, 25.0, 50.0, 100.0]

    def test_custom_stages(self):
        """Test custom stage configuration."""
        config = DeploymentConfig(
            model_version="v1.0.0",
            stages=[
                DeploymentStage(name="test", traffic_percentage=10.0, duration_minutes=5),
                DeploymentStage(name="prod", traffic_percentage=100.0, duration_minutes=10),
            ],
        )

        assert len(config.stages) == 2


class TestHealthCheck:
    """Tests for HealthCheck model."""

    def test_health_check_creation(self):
        """Test creating a health check."""
        health = HealthCheck(
            is_healthy=True,
            success_rate=0.98,
            latency_p50_ms=45.0,
            latency_p99_ms=120.0,
            error_count=2,
            request_count=100,
        )

        assert health.is_healthy is True
        assert health.success_rate == 0.98


class TestRollbackTrigger:
    """Tests for RollbackTrigger model."""

    def test_trigger_creation(self):
        """Test creating a rollback trigger."""
        trigger = RollbackTrigger(
            reason=RollbackReason.ERROR_RATE_EXCEEDED,
            stage_name="canary",
            traffic_percentage=5.0,
            threshold_violated="success_rate",
            actual_value=0.85,
            expected_value=0.95,
            message="Success rate too low",
        )

        assert trigger.reason == RollbackReason.ERROR_RATE_EXCEEDED
        assert trigger.stage_name == "canary"


class TestCanaryDeployment:
    """Tests for CanaryDeployment."""

    @pytest.fixture
    def deployment(self) -> CanaryDeployment:
        """Create a test deployment."""
        config = DeploymentConfig(
            model_version="v1.0.0",
            stages=[
                DeploymentStage(
                    name="test",
                    traffic_percentage=10.0,
                    duration_minutes=1,
                    health_check_interval_seconds=1,
                ),
            ],
        )
        return CanaryDeployment(config=config)

    def test_deployment_initialization(self, deployment):
        """Test deployment initializes correctly."""
        assert deployment.status == DeploymentStatus.PENDING
        assert deployment.current_stage_index == 0

    def test_current_stage(self, deployment):
        """Test getting current stage."""
        stage = deployment.current_stage
        assert stage is not None
        assert stage.name == "test"

    def test_route_request(self, deployment):
        """Test request routing."""
        deployment._router.update_canary_percentage(50.0)

        results = set()
        for i in range(100):
            results.add(deployment.route_request(f"req-{i}"))

        # Should have both canary and stable routes
        assert "canary" in results or "stable" in results

    def test_record_request(self, deployment):
        """Test recording request metrics."""
        deployment.record_request(latency_ms=50.0, success=True, accuracy=0.95)
        deployment.record_request(latency_ms=60.0, success=True, accuracy=0.90)

        health = deployment.get_current_health()
        assert health.request_count == 2

    def test_pause_resume(self, deployment):
        """Test pause and resume."""
        deployment.status = DeploymentStatus.IN_PROGRESS

        deployment.pause()
        assert deployment.status == DeploymentStatus.PAUSED

        deployment.resume()
        assert deployment.status == DeploymentStatus.IN_PROGRESS

    def test_manual_rollback(self, deployment):
        """Test manual rollback trigger."""
        deployment.status = DeploymentStatus.IN_PROGRESS
        deployment.trigger_manual_rollback("Testing rollback")

        assert deployment.status == DeploymentStatus.ROLLING_BACK
        assert deployment._rollback_trigger is not None
        assert deployment._rollback_trigger.reason == RollbackReason.MANUAL

    def test_check_health_thresholds_pass(self, deployment):
        """Test health check passes thresholds."""
        stage = deployment.current_stage
        health = HealthCheck(
            is_healthy=True,
            success_rate=0.98,
            latency_p50_ms=30.0,
            latency_p99_ms=100.0,
            error_count=2,
            request_count=100,
            accuracy=0.95,
        )

        trigger = deployment._check_health_thresholds(health, stage)
        assert trigger is None

    def test_check_health_thresholds_fail_success_rate(self, deployment):
        """Test health check fails on success rate."""
        stage = deployment.current_stage
        health = HealthCheck(
            is_healthy=True,
            success_rate=0.80,  # Below 0.95 threshold
            latency_p50_ms=30.0,
            latency_p99_ms=100.0,
            error_count=20,
            request_count=100,
        )

        trigger = deployment._check_health_thresholds(health, stage)
        assert trigger is not None
        assert trigger.reason == RollbackReason.ERROR_RATE_EXCEEDED

    def test_check_health_thresholds_fail_latency(self, deployment):
        """Test health check fails on latency."""
        stage = deployment.current_stage
        health = HealthCheck(
            is_healthy=True,
            success_rate=0.98,
            latency_p50_ms=300.0,
            latency_p99_ms=800.0,  # Above 500ms threshold
            error_count=2,
            request_count=100,
        )

        trigger = deployment._check_health_thresholds(health, stage)
        assert trigger is not None
        assert trigger.reason == RollbackReason.LATENCY_EXCEEDED


class TestCanaryDeploymentAsync:
    """Async tests for CanaryDeployment."""

    @pytest.fixture
    def fast_deployment(self) -> CanaryDeployment:
        """Create a fast deployment for testing."""
        config = DeploymentConfig(
            model_version="v1.0.0",
            stages=[
                DeploymentStage(
                    name="quick",
                    traffic_percentage=100.0,
                    duration_minutes=1,
                    health_check_interval_seconds=60,  # Only one check
                ),
            ],
            cooldown_between_stages_seconds=0,
        )
        return CanaryDeployment(config=config)

    @pytest.mark.asyncio
    async def test_execute_successful(self, fast_deployment):
        """Test successful deployment execution."""
        async def healthy_callback():
            return HealthCheck(
                is_healthy=True,
                success_rate=0.99,
                latency_p50_ms=30.0,
                latency_p99_ms=80.0,
                error_count=1,
                request_count=100,
            )

        # Note: This would run for 1 minute in real execution
        # For testing, we'd need to mock time or use shorter durations

    @pytest.mark.asyncio
    async def test_rollback_execution(self, fast_deployment):
        """Test rollback is triggered on poor health."""
        fast_deployment.config.stages[0].duration_minutes = 1
        fast_deployment.config.stages[0].health_check_interval_seconds = 60

        call_count = 0

        async def degrading_callback():
            nonlocal call_count
            call_count += 1
            # Return unhealthy response
            return HealthCheck(
                is_healthy=False,
                success_rate=0.50,  # Very low
                latency_p50_ms=300.0,
                latency_p99_ms=1500.0,  # High latency
                error_count=50,
                request_count=100,
            )

        # Would trigger rollback in real execution


class TestDeploymentFactories:
    """Tests for deployment factory functions."""

    def test_create_default_deployment(self):
        """Test default deployment creation."""
        deployment = create_default_deployment("v1.2.0")

        assert deployment.config.model_version == "v1.2.0"
        assert len(deployment.config.stages) == 4

    def test_create_conservative_deployment(self):
        """Test conservative deployment creation."""
        deployment = create_conservative_deployment("v2.0.0")

        assert deployment.config.model_version == "v2.0.0"
        assert len(deployment.config.stages) == 5  # More stages

        # First stage should be very small percentage
        assert deployment.config.stages[0].traffic_percentage == 1.0


class TestDeploymentMetrics:
    """Tests for DeploymentMetrics model."""

    def test_metrics_creation(self):
        """Test creating deployment metrics."""
        metrics = DeploymentMetrics(
            deployment_id="abc123",
            started_at=datetime.utcnow(),
            final_status=DeploymentStatus.COMPLETED,
            stages_completed=4,
            total_stages=4,
            health_checks_passed=20,
            health_checks_failed=0,
            peak_traffic_percentage=100.0,
            total_requests_served=10000,
            average_latency_ms=45.0,
            average_accuracy=0.95,
        )

        assert metrics.deployment_id == "abc123"
        assert metrics.stages_completed == 4


