"""Pytest configuration and fixtures for EdgeDeploy tests."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

# Seed for reproducibility
SEED = 42


def set_deterministic_seed(seed: int = SEED) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture(autouse=True)
def reset_seeds() -> None:
    """Reset seeds before each test."""
    set_deterministic_seed()


class SimpleCNN(nn.Module):
    """Simple CNN for testing quantization and export."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class SimpleMLP(nn.Module):
    """Simple MLP for testing."""

    def __init__(self, input_dim: int = 100, num_classes: int = 10) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@pytest.fixture
def simple_cnn() -> SimpleCNN:
    """Create a simple CNN model."""
    set_deterministic_seed()
    model = SimpleCNN(num_classes=10)
    model.eval()
    return model


@pytest.fixture
def simple_mlp() -> SimpleMLP:
    """Create a simple MLP model."""
    set_deterministic_seed()
    model = SimpleMLP(input_dim=100, num_classes=10)
    model.eval()
    return model


@pytest.fixture
def sample_image_batch() -> torch.Tensor:
    """Create a sample batch of images."""
    set_deterministic_seed()
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def sample_vector_batch() -> torch.Tensor:
    """Create a sample batch of vectors."""
    set_deterministic_seed()
    return torch.randn(4, 100)


@pytest.fixture
def mock_dataloader() -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create a mock dataloader (list of batches)."""
    set_deterministic_seed()
    batches = []
    for _ in range(10):
        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 10, (8,))
        batches.append((images, labels))
    return batches


@pytest.fixture
def reference_distribution() -> np.ndarray:
    """Create reference distribution for drift testing."""
    set_deterministic_seed()
    return np.random.normal(0, 1, size=1000)


@pytest.fixture
def shifted_distribution() -> np.ndarray:
    """Create shifted distribution for drift testing."""
    set_deterministic_seed()
    return np.random.normal(0.5, 1.2, size=1000)


@pytest.fixture
def no_drift_distribution() -> np.ndarray:
    """Create distribution similar to reference (no drift)."""
    set_deterministic_seed()
    # Use different random state but same parameters
    rng = np.random.RandomState(43)
    return rng.normal(0, 1, size=1000)


