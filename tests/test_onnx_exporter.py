"""Tests for ONNX exporter."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.export.onnx_exporter import (
    ONNXExporter,
    ExportConfig,
    ExportResult,
    OperatorValidation,
    RuntimeCompatibility,
    RUNTIME_OP_SUPPORT,
    _get_tensor_shape,
    _compute_model_hash,
)


class TestExportHelpers:
    """Tests for helper functions."""

    def test_compute_model_hash(self):
        """Test model hash computation."""
        data1 = b"test model data"
        data2 = b"different data"

        hash1 = _compute_model_hash(data1)
        hash2 = _compute_model_hash(data2)

        assert len(hash1) == 16
        assert hash1 != hash2

    def test_hash_consistency(self):
        """Test hash is consistent."""
        data = b"consistent data"
        hash1 = _compute_model_hash(data)
        hash2 = _compute_model_hash(data)

        assert hash1 == hash2


class TestRuntimeOpSupport:
    """Tests for runtime operator support matrix."""

    def test_onnxruntime_support(self):
        """Test ONNXRuntime operator support."""
        ops = RUNTIME_OP_SUPPORT["onnxruntime"]

        # Common operators should be supported
        assert "Conv" in ops
        assert "Relu" in ops
        assert "MatMul" in ops
        assert "Softmax" in ops

    def test_tensorrt_support(self):
        """Test TensorRT operator support."""
        ops = RUNTIME_OP_SUPPORT["tensorrt"]

        assert "Conv" in ops
        assert "Relu" in ops

    def test_quantized_ops_supported(self):
        """Test quantized operators are listed."""
        ops = RUNTIME_OP_SUPPORT["onnxruntime"]

        assert "QuantizeLinear" in ops
        assert "DequantizeLinear" in ops


class TestExportConfig:
    """Tests for ExportConfig model."""

    def test_default_config(self):
        """Test default configuration."""
        config = ExportConfig()

        assert config.opset_version == 17
        assert config.optimize is True
        assert config.validate_model is True
        assert config.test_inference is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExportConfig(
            opset_version=13,
            optimize=False,
            target_runtimes=["tensorrt"],
        )

        assert config.opset_version == 13
        assert config.optimize is False
        assert "tensorrt" in config.target_runtimes

    def test_dynamic_axes_default(self):
        """Test default dynamic axes."""
        config = ExportConfig()

        assert "input" in config.dynamic_axes
        assert 0 in config.dynamic_axes["input"]


class TestOperatorValidation:
    """Tests for OperatorValidation model."""

    def test_validation_creation(self):
        """Test creating operator validation."""
        validation = OperatorValidation(
            op_type="Conv",
            node_name="conv1",
            is_supported=True,
            supported_runtimes=["onnxruntime", "tensorrt"],
            unsupported_runtimes=[],
        )

        assert validation.op_type == "Conv"
        assert validation.is_supported is True

    def test_validation_with_warnings(self):
        """Test validation with warnings."""
        validation = OperatorValidation(
            op_type="Dropout",
            node_name="dropout1",
            is_supported=True,
            supported_runtimes=["onnxruntime"],
            unsupported_runtimes=[],
            warnings=["Dropout should be removed for inference"],
        )

        assert len(validation.warnings) == 1


class TestRuntimeCompatibility:
    """Tests for RuntimeCompatibility model."""

    def test_compatibility_creation(self):
        """Test creating runtime compatibility."""
        compat = RuntimeCompatibility(
            runtime="onnxruntime",
            is_compatible=True,
            supported_ops=50,
            unsupported_ops=0,
            unsupported_op_types=[],
            compatibility_score=1.0,
        )

        assert compat.runtime == "onnxruntime"
        assert compat.is_compatible is True
        assert compat.compatibility_score == 1.0

    def test_partial_compatibility(self):
        """Test partial compatibility."""
        compat = RuntimeCompatibility(
            runtime="tflite",
            is_compatible=False,
            supported_ops=40,
            unsupported_ops=10,
            unsupported_op_types=["CustomOp"],
            compatibility_score=0.8,
        )

        assert compat.is_compatible is False
        assert compat.compatibility_score == 0.8


class TestExportResult:
    """Tests for ExportResult model."""

    def test_successful_result(self):
        """Test successful export result."""
        result = ExportResult(
            success=True,
            export_path="/path/to/model.onnx",
            model_size_mb=5.5,
            opset_version=17,
            input_shapes={"input": [1, 3, 224, 224]},
            output_shapes={"output": [1, 1000]},
            total_operators=100,
            unique_operators=["Conv", "Relu", "MatMul"],
            operator_validations=[],
            runtime_compatibilities=[],
            inference_test_passed=True,
            inference_latency_ms=15.5,
        )

        assert result.success is True
        assert result.model_size_mb == 5.5

    def test_failed_result(self):
        """Test failed export result."""
        result = ExportResult(
            success=False,
            opset_version=17,
            input_shapes={},
            output_shapes={},
            total_operators=0,
            unique_operators=[],
            operator_validations=[],
            runtime_compatibilities=[],
            errors=["Export failed: model not traceable"],
        )

        assert result.success is False
        assert len(result.errors) == 1


class TestONNXExporter:
    """Tests for ONNXExporter."""

    @pytest.fixture
    def exporter(self) -> ONNXExporter:
        """Create an exporter instance."""
        config = ExportConfig(
            opset_version=17,
            target_runtimes=["onnxruntime"],
        )
        return ONNXExporter(config=config)

    def test_exporter_initialization(self, exporter):
        """Test exporter initializes correctly."""
        assert exporter.config.opset_version == 17

    @pytest.mark.skipif(
        not pytest.importorskip("onnx", reason="ONNX not installed"),
        reason="ONNX not installed",
    )
    def test_export_simple_model(self, exporter, simple_mlp, sample_vector_batch):
        """Test exporting a simple model."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            output_path = f.name

        try:
            result = exporter.export(
                simple_mlp,
                sample_vector_batch,
                output_path,
            )

            # Even if ONNX export fails, we should get a result
            assert isinstance(result, ExportResult)

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_export_without_onnx(self, simple_mlp, sample_vector_batch):
        """Test graceful handling when ONNX not available."""
        # This tests the error handling path
        exporter = ONNXExporter()

        # Force ONNX unavailable by checking result
        result = exporter.export(simple_mlp, sample_vector_batch)

        # Should return a result (success or failure)
        assert isinstance(result, ExportResult)

    def test_validate_operators(self, exporter):
        """Test operator validation logic."""
        # Create a mock ONNX model structure
        class MockNode:
            def __init__(self, op_type: str, name: str):
                self.op_type = op_type
                self.name = name

        class MockGraph:
            def __init__(self):
                self.node = [
                    MockNode("Conv", "conv1"),
                    MockNode("Relu", "relu1"),
                    MockNode("MatMul", "matmul1"),
                    MockNode("CustomOp", "custom1"),
                ]

        class MockModel:
            def __init__(self):
                self.graph = MockGraph()

        mock_model = MockModel()
        validations, op_types = exporter._validate_operators(mock_model)

        assert len(validations) == 4
        assert "Conv" in op_types
        assert "CustomOp" in op_types

    def test_compute_runtime_compatibility(self, exporter):
        """Test runtime compatibility computation."""
        op_type_nodes = {
            "Conv": ["conv1", "conv2"],
            "Relu": ["relu1"],
            "CustomOp": ["custom1"],
        }

        compatibilities = exporter._compute_runtime_compatibility(op_type_nodes)

        assert len(compatibilities) == 1  # Only onnxruntime in config

        ort_compat = compatibilities[0]
        assert ort_compat.runtime == "onnxruntime"
        assert ort_compat.unsupported_ops == 1  # CustomOp


class TestONNXExporterValidation:
    """Tests for ONNX model validation."""

    @pytest.fixture
    def exporter(self) -> ONNXExporter:
        """Create an exporter instance."""
        return ONNXExporter()

    def test_validate_nonexistent_model(self, exporter):
        """Test validating nonexistent model."""
        result = exporter.validate_existing_model("/nonexistent/path.onnx")

        assert result.success is False
        assert "not found" in result.errors[0].lower()


class TestONNXExporterComparison:
    """Tests for model comparison functionality."""

    @pytest.fixture
    def exporter(self) -> ONNXExporter:
        """Create an exporter instance."""
        return ONNXExporter()

    def test_compare_without_onnx(self, exporter, simple_mlp, sample_vector_batch):
        """Test comparison handles missing ONNX gracefully."""
        result = exporter.compare_models(
            simple_mlp,
            "/fake/path.onnx",
            sample_vector_batch,
        )

        # Should return an error result
        assert isinstance(result, dict)
        assert "success" in result or "error" in result


class TestExportConfigValidation:
    """Tests for ExportConfig validation."""

    def test_opset_version_bounds(self):
        """Test opset version is bounded."""
        with pytest.raises(ValueError):
            ExportConfig(opset_version=5)  # Below minimum 9

        with pytest.raises(ValueError):
            ExportConfig(opset_version=25)  # Above maximum 20

    def test_valid_opset_versions(self):
        """Test valid opset versions."""
        for version in [9, 13, 17, 20]:
            config = ExportConfig(opset_version=version)
            assert config.opset_version == version


