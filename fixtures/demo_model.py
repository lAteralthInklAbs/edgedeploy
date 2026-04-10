"""Demo models for testing and evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn


class DemoVisionModel(nn.Module):
    """
    Demo vision model for edge deployment testing.

    A lightweight CNN suitable for automotive perception tasks.
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class DemoObjectDetector(nn.Module):
    """
    Simplified object detector backbone for testing.

    Outputs feature maps suitable for detection heads.
    """

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            # Stage 1: 1/2
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Stage 2: 1/4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Stage 3: 1/8
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Stage 4: 1/16
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 100),  # Simplified output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        output = self.head(features)
        return output


class DemoTimeSeriesModel(nn.Module):
    """
    Demo time series model for sensor data processing.
    """

    def __init__(
        self,
        input_features: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 5,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last timestep
        last_output = lstm_out[:, -1, :]
        return self.classifier(last_output)


def create_demo_model(
    model_type: str = "vision",
    num_classes: int = 10,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create demo models.

    Args:
        model_type: One of "vision", "detector", "timeseries"
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments

    Returns:
        Instantiated model
    """
    models = {
        "vision": DemoVisionModel,
        "detector": DemoObjectDetector,
        "timeseries": DemoTimeSeriesModel,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    model_class = models[model_type]

    if model_type in ("vision", "timeseries"):
        return model_class(num_classes=num_classes, **kwargs)
    else:
        return model_class(**kwargs)


def create_sample_input(
    model_type: str = "vision",
    batch_size: int = 1,
) -> torch.Tensor:
    """
    Create sample input tensor for a model type.

    Args:
        model_type: One of "vision", "detector", "timeseries"
        batch_size: Batch size

    Returns:
        Sample input tensor
    """
    if model_type in ("vision", "detector"):
        return torch.randn(batch_size, 3, 224, 224)
    elif model_type == "timeseries":
        return torch.randn(batch_size, 50, 10)  # 50 timesteps, 10 features
    else:
        raise ValueError(f"Unknown model type: {model_type}")


