"""Tests for quantization engine."""

from __future__ import annotations

import pytest
import torch

from src.optimization.quantization_engine import (
    QuantizationEngine,
    QuantizationConfig,
    QuantizationResult,
    LayerSensitivity,
    QuantizableWrapper,
    _count_parameters,
    _estimate_model_size_mb,
    _get_layer_names,
)


class TestQuantizationHelpers:
    """Tests for helper functions."""

    def test_count_parameters(self, simple_cnn):
        """Test parameter counting."""
        params = _count_parameters(simple_cnn)
        assert params > 0
        # Simple CNN should have a reasonable number of parameters
        assert 100_000 < params < 1_000_000

    def test_estimate_model_size_fp32(self, simple_cnn):
        """Test model size estimation in FP32."""
        size_mb = _estimate_model_size_mb(simple_cnn, dtype_bits=32)
        assert size_mb > 0
        # FP32 model should be ~1-3MB for our simple CNN
        assert 0.1 < size_mb < 10.0

    def test_estimate_model_size_int8(self, simple_cnn):
        """Test model size estimation in INT8."""
        size_fp32 = _estimate_model_size_mb(simple_cnn, dtype_bits=32)
        size_int8 = _estimate_model_size_mb(simple_cnn, dtype_bits=8)
        # INT8 should be ~4x smaller
        assert abs(size_fp32 / size_int8 - 4.0) < 0.1

    def test_get_layer_names(self, simple_cnn):
        """Test getting quantizable layer names."""
        layers = _get_layer_names(simple_cnn)
        assert len(layers) > 0

        # Check we found Conv2d and Linear layers
        layer_types = [type(module).__name__ for _, module in layers]
        assert "Conv2d" in layer_types
        assert "Linear" in layer_types


class TestQuantizableWrapper:
    """Tests for QuantizableWrapper."""

    def test_wrapper_forward(self, simple_cnn, sample_image_batch):
        """Test wrapped model forward pass."""
        wrapped = QuantizableWrapper(simple_cnn)
        output = wrapped(sample_image_batch)
        assert output.shape == (4, 10)

    def test_wrapper_preserves_output(self, simple_cnn, sample_image_batch):
        """Test wrapper doesn't change output values."""
        wrapped = QuantizableWrapper(simple_cnn)

        with torch.no_grad():
            original_output = simple_cnn(sample_image_batch)
            wrapped_output = wrapped(sample_image_batch)

        # Outputs should be identical (wrapper just adds stubs)
        torch.testing.assert_close(original_output, wrapped_output)


class TestQuantizationConfig:
    """Tests for QuantizationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QuantizationConfig()
        assert config.qat_epochs == 5
        assert config.calibration_batches == 100
        assert config.sensitivity_threshold == 0.02
        assert config.target_compression == 4.0
        assert config.accuracy_retention_target == 0.97

    def test_custom_config(self):
        """Test custom configuration."""
        config = QuantizationConfig(
            qat_epochs=10,
            sensitivity_threshold=0.05,
        )
        assert config.qat_epochs == 10
        assert config.sensitivity_threshold == 0.05

    def test_config_validation(self):
        """Test config validation rejects invalid values."""
        with pytest.raises(ValueError):
            QuantizationConfig(qat_epochs=0)  # Must be >= 1

        with pytest.raises(ValueError):
            QuantizationConfig(sensitivity_threshold=-0.1)  # Must be >= 0


class TestQuantizationEngine:
    """Tests for QuantizationEngine."""

    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        engine = QuantizationEngine()
        assert engine.config is not None
        assert engine.device == "cpu"

    def test_engine_with_custom_config(self):
        """Test engine with custom configuration."""
        config = QuantizationConfig(qat_epochs=3)
        engine = QuantizationEngine(config=config, device="cpu")
        assert engine.config.qat_epochs == 3

    def test_evaluate_accuracy(self, simple_cnn, mock_dataloader):
        """Test accuracy evaluation."""
        engine = QuantizationEngine()
        accuracy = engine._evaluate_accuracy(
            simple_cnn,
            mock_dataloader,
            num_batches=5,
        )
        # Accuracy should be between 0 and 1
        assert 0.0 <= accuracy <= 1.0

    def test_analyze_layer_sensitivity(self, simple_cnn, mock_dataloader):
        """Test layer sensitivity analysis."""
        engine = QuantizationEngine()
        sensitivities = engine.analyze_layer_sensitivity(
            simple_cnn,
            mock_dataloader,
            calibration_batches=3,
        )

        assert len(sensitivities) > 0

        for sens in sensitivities:
            assert isinstance(sens, LayerSensitivity)
            assert 0.0 <= sens.sensitivity_score <= 1.0
            assert sens.recommended_precision in ("INT8", "FP16", "FP32")

    def test_compute_mixed_precision_config(self, simple_cnn, mock_dataloader):
        """Test mixed precision config computation."""
        engine = QuantizationEngine()
        sensitivities = engine.analyze_layer_sensitivity(
            simple_cnn,
            mock_dataloader,
            calibration_batches=2,
        )

        config = engine.compute_mixed_precision_config(sensitivities)

        assert isinstance(config, dict)
        assert len(config) == len(sensitivities)
        for precision in config.values():
            assert precision in ("INT8", "FP16", "FP32")

    def test_compute_pareto_frontier(self, simple_cnn, mock_dataloader):
        """Test Pareto frontier computation."""
        engine = QuantizationEngine()
        sensitivities = engine.analyze_layer_sensitivity(
            simple_cnn,
            mock_dataloader,
            calibration_batches=2,
        )

        pareto = engine.compute_pareto_frontier(sensitivities, num_points=5)

        assert len(pareto) == 5

        for point in pareto:
            assert "compression_ratio" in point
            assert "accuracy_retention" in point
            assert "layers_quantized" in point
            assert point["compression_ratio"] >= 1.0
            assert 0.0 <= point["accuracy_retention"] <= 1.0


class TestLayerSensitivity:
    """Tests for LayerSensitivity model."""

    def test_layer_sensitivity_creation(self):
        """Test creating LayerSensitivity."""
        sens = LayerSensitivity(
            layer_name="conv1",
            layer_type="Conv2d",
            original_accuracy=0.95,
            quantized_accuracy=0.93,
            accuracy_drop=0.02,
            sensitivity_score=0.2,
            recommended_precision="INT8",
            num_parameters=1000,
        )

        assert sens.layer_name == "conv1"
        assert sens.sensitivity_score == 0.2

    def test_sensitivity_score_validation(self):
        """Test sensitivity score must be in [0, 1]."""
        with pytest.raises(ValueError):
            LayerSensitivity(
                layer_name="conv1",
                layer_type="Conv2d",
                original_accuracy=0.95,
                quantized_accuracy=0.93,
                accuracy_drop=0.02,
                sensitivity_score=1.5,  # Invalid
                recommended_precision="INT8",
                num_parameters=1000,
            )


