#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Quantize TinyCNN for Ethos-U and save TinyCNN.pt2 alongside this script.

Run using the repository's Python environment:

    .venv/Scripts/python.exe model/TinyCNN/quantize_model.py

This standalone example defines its own network and calibration data. It uses
the same symmetric INT8, per-channel-weight PT2E quantization as create_ai_layer.py
and saves the quantized ``forward`` program before backend lowering. Vela runs
later, when create_ai_layer.py consumes the .pt2 archive.
"""

from pathlib import Path

import torch
from torch import nn
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
)
from torch.export.pt2_archive._package import package_pt2
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


BATCH_SIZE = 1
IMAGE_HEIGHT = 16
IMAGE_WIDTH = 16
INPUT_CHANNELS = 3
FIRST_CONV_CHANNELS = 8
SECOND_CONV_CHANNELS = 16
POOL_SIZE = 2
NUM_CLASSES = 10

INPUT_SHAPE = (BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, INPUT_CHANNELS)  # NHWC
OUTPUT_PATH = Path(__file__).with_name("TinyCNN.pt2")


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, FIRST_CONV_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(POOL_SIZE),
            nn.Conv2d(FIRST_CONV_CHANNELS, SECOND_CONV_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(POOL_SIZE),
        )
        # Both convolutions preserve spatial dimensions; each pool downsamples once.
        feature_height = IMAGE_HEIGHT // POOL_SIZE // POOL_SIZE
        feature_width = IMAGE_WIDTH // POOL_SIZE // POOL_SIZE
        self.classifier = nn.Linear(
            SECOND_CONV_CHANNELS * feature_height * feature_width, NUM_CLASSES
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x.permute(0, 3, 1, 2))
        return self.classifier(torch.flatten(x, 1))


def quantize_model() -> Path:
    """Save the quantized TinyCNN checkpoint and return its path."""
    # Match model.py's deterministic, untrained network and calibration inputs.
    torch.manual_seed(0)
    model = TinyCNN().eval()
    example_args = (torch.zeros(INPUT_SHAPE),)
    samples = [torch.full(INPUT_SHAPE, 0.5), torch.full(INPUT_SHAPE, -0.5)]
    graph = torch.export.export(model, example_args).module(check_guards=False)

    # This specification selects quantization rules without invoking Vela.
    quantizer = EthosUQuantizer(EthosUCompileSpec(target="ethos-u55-256"))
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(graph, quantizer)
    with torch.no_grad():
        for sample in samples:
            prepared(sample)
    quantized = convert_pt2e(prepared)
    exported = torch.export.export(quantized, example_args)

    package_pt2(OUTPUT_PATH, exported_programs={"forward": exported})
    return OUTPUT_PATH


if __name__ == "__main__":
    output = quantize_model()
    print(f"[TinyCNN] wrote {output}")
