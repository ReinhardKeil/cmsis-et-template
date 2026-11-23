#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Save filter and classify in TinyCNNMultiple.pt2, before backend lowering.

Run with the repository's Python environment. No CMSIS tooling or Vela is
called here. create_ai_layer.py later preserves the named networks and turns
constant_methods.json into ExecuTorch constant methods.
"""

import json
from pathlib import Path

import torch
from torch import nn
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_a16w8_quantization_config,
    get_symmetric_quantization_config,
)
from torch.export.pt2_archive._package import package_pt2
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


IMAGE_HEIGHT = 16
IMAGE_WIDTH = 16
INPUT_CHANNELS = 3
FIRST_CONV_CHANNELS = 8
SECOND_CONV_CHANNELS = 16
POOL_SIZE = 2
NUM_CLASSES = 10
# Inference always uses one image.
INPUT_SHAPE = (1, IMAGE_HEIGHT, IMAGE_WIDTH, INPUT_CHANNELS)
OUTPUT_PATH = Path(__file__).with_name("TinyCNNMultiple.pt2")


class ImageFilter(nn.Module):
    """Independent 3x3 box filter for each image channel; NHWC in and out."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            INPUT_CHANNELS, INPUT_CHANNELS, kernel_size=3, padding=1,
            groups=INPUT_CHANNELS, bias=False,
        )
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / 9.0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.conv(image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, FIRST_CONV_CHANNELS, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(POOL_SIZE),
            nn.Conv2d(FIRST_CONV_CHANNELS, SECOND_CONV_CHANNELS, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(POOL_SIZE),
        )
        self.classifier = nn.Linear(
            SECOND_CONV_CHANNELS * (IMAGE_HEIGHT // POOL_SIZE // POOL_SIZE)
            * (IMAGE_WIDTH // POOL_SIZE // POOL_SIZE), NUM_CLASSES,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.features(image.permute(0, 3, 1, 2)), 1))


def quantize_network(model: nn.Module, samples: list[torch.Tensor], config):
    example = (torch.zeros(INPUT_SHAPE),)
    graph = torch.export.export(model, example).module(check_guards=False)
    quantizer = EthosUQuantizer(EthosUCompileSpec(target="ethos-u55-256"))
    quantizer.set_global(config)
    # Cortex-M's quantized_linear requires one symmetric weight zero point.
    # Keep convolution weights per-channel and use per-tensor linear weights
    # so this single checkpoint can also be lowered for CPU-only projects.
    quantizer.set_module_type(nn.Linear, get_symmetric_quantization_config(is_per_channel=False))
    prepared = prepare_pt2e(graph, quantizer)
    with torch.no_grad():
        for sample in samples:
            prepared(sample)
    return torch.export.export(convert_pt2e(prepared), example)


def quantize_models() -> Path:
    torch.manual_seed(0)
    classifier = TinyCNN().eval()
    image_filter = ImageFilter().eval()
    samples = [torch.full(INPUT_SHAPE, 0.5), torch.full(INPUT_SHAPE, -0.5)]
    filter_program = quantize_network(
        image_filter, samples, get_symmetric_a16w8_quantization_config(is_per_channel=True)
    )
    quantized_filter = filter_program.module(check_guards=False)
    with torch.no_grad():
        classifier_samples = [quantized_filter(sample) for sample in samples]
    classify_program = quantize_network(
        classifier, classifier_samples, get_symmetric_quantization_config(is_per_channel=True)
    )
    constant_methods = {
        "IMAGE_WIDTH": IMAGE_WIDTH,
        "IMAGE_HEIGHT": IMAGE_HEIGHT,
        "INPUT_CHANNELS": INPUT_CHANNELS,
        "NUM_CLASSES": NUM_CLASSES,
    }
    package_pt2(
        OUTPUT_PATH,
        exported_programs={"filter": filter_program, "classify": classify_program},
        extra_files={"constant_methods.json": json.dumps(constant_methods)},
    )
    with torch.no_grad():
        filtered = quantized_filter(torch.full(INPUT_SHAPE, 0.25))
        reference = classify_program.module(check_guards=False)(filtered)
    print("[TinyCNNMultiple] reference logits:", reference.flatten().tolist())
    return OUTPUT_PATH


if __name__ == "__main__":
    print(f"[TinyCNNMultiple] wrote {quantize_models()}")
