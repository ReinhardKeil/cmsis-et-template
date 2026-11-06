# Create ExecuTorch applications for real devices

An Edge AI application is more than an ML model. The model must fit the
target's memory architecture and optional Ethos-U configuration, connect to
the application and its sensors, and run reliably on the real device.

This guide explains how to turn one of this repository's template applications
into a hardware-specific application. It follows the development loop described
in [Making Edge AI development more productive](https://developer.arm.com/community/arm-community-blogs/b/tools-software-ides-blog/posts/making-edge-ai-development-more-productive):

1. describe the target accurately;
2. generate an optimized model and AI software layer;
3. build and validate the complete application on an FVP;
4. deploy the same application to hardware;
5. collect results and real data for the next iteration.

This document is adapted for this template repository from Arm's
[Create ExecuTorch applications for real devices](https://github.com/Arm-Examples/.github/blob/main/profile/executorch.md)
guide.

## Start with the matching template

Select the directory closest to the intended software environment and
processor:

| Directory | Software environment | ML execution |
| --- | --- | --- |
| [`cmsis-u55/`](../cmsis-u55/README.md) | CMSIS | Cortex-M with Ethos-U55/U65 |
| [`cmsis-u85/`](../cmsis-u85/README.md) | CMSIS | Cortex-M with Ethos-U85 |
| `cmsis-m/` | CMSIS | Cortex-M only |
| `zephyr-u55/` | Zephyr | Cortex-M with Ethos-U55/U65 |
| [`zephyr-u85/`](../zephyr-u85/README.md) | Zephyr | Cortex-M with Ethos-U85 |
| `zephyr-m/` | Zephyr | Cortex-M only |

Begin with the template's Quick Start and run the supplied reference target
before changing it. This separates tool and environment problems from issues
introduced while adapting the application.

## How the pieces work together

[CMSIS-Toolbox](https://open-cmsis-pack.github.io/cmsis-toolbox/) resolves the
selected target-set and the solution's `mlops:` information into a generated
`*.cbuild-mlops.yml` file. The model-conversion tools consume that description,
quantize and export the PyTorch model, delegate supported operators to the
selected backend, and create the target-specific AI layer. The final build
combines that layer with the embedded application.

For Ethos-U targets, the NPU type, MAC count, Vela settings, memory layout,
driver configuration, and FVP must all describe the same hardware. Follow the
[CMSIS-Ethos-U integration guide](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html)
when changing these settings. The CMSIS templates use the
`ARM::CMSIS-Ethos-U` pack; the Zephyr templates instead use the Ethos-U driver
included in upstream Zephyr.

See [MLOps flow](mlops-flow.md) for the generated files and the commands behind
the three-step setup, AI-layer generation, and application-build workflow.

## Adapt the application

Replace the starter model and calibration inputs with data representative of
the intended product. Then adapt the embedded input path, tensor shapes and
types, output interpretation, and application logic. A research model often
needs changes before it quantizes well or fits the target; check both accuracy
and memory use rather than assuming an unchanged export is suitable.

Keep hardware-specific information in the target description instead of
duplicating it in the Python exporter. For a new board, supply the appropriate
device support, memory regions, cache policy, interrupt routing, debugger, and
run configuration. When no packaged board support fits, create a board layer
or Zephyr board/overlay configuration for that hardware.

## Validate on an FVP first

An FVP allows application integration to begin before hardware is available.
Build a small self-test that uses a fixed input and compares the target result
with a host-side PyTorch reference. A useful FVP run reports at least:

- the processor and optional NPU configuration;
- model and tensor shapes;
- inference output and a clear pass/fail result;
- model, method-pool, temporary-pool, and application-buffer sizes;
- which memory region holds each large object.

Use the same NPU and memory arrangement as the hardware target. A Corstone
reference memory map is useful for initial integration, but it is not a
substitute for the real device's memory and Vela configuration.

FVP execution is slower than hardware. Keep simulator validation deterministic
and small, then run larger data sets on the board or in host-side tests.

## Run the same application on hardware

Add the board target without changing the model contract or self-test. Build,
program, and run the application with the board's supported debugger and
console. Prove the fixed-input self-test before enabling live camera,
microphone, or sensor pipelines; startup, memory, interrupt, and cache problems
are easier to isolate without continuously changing input data.

For an Ethos-U application, an FVP and board using the same command stream
should produce identical inference output. A difference usually points to
integration details such as memory placement, cache maintenance, NPU access,
or interrupt routing rather than to model conversion.

The
[CMSIS Developer Assistant](https://marketplace.visualstudio.com/items?itemName=Arm.cmsis-developer-assistant)
can inspect CMSIS project information, build artefacts, registers, and memory
through the CMSIS Debugger. It is particularly useful when a failure appears
only on the physical target.

## Make real-world inputs repeatable

Sensor, audio, and video inputs change continuously. Capture representative
inputs and replay the same data on the FVP and hardware so that a field failure
becomes a repeatable regression test. The
[Synchronous Data Stream Framework](https://github.com/ARM-software/SDS-Framework)
can transport and record these streams.

Define one stream for each boundary that a test must control or observe. For a
vision application, for example:

| Stream | Direction | Contents |
| --- | --- | --- |
| Input | Playback to application | The frame presented at the model boundary |
| Model output | Record from application | Raw logits, heatmaps, boxes, or keypoints |
| Result | Record from application | Post-processed result used by the product |

Keep raw model output separate from post-processed application output. This
makes it possible to tell whether a regression came from model conversion or
from embedded post-processing.

Use task-specific tolerances when comparing a quantized target with the
floating-point host model. Expect identical results when comparing the same
quantized program on FVP and hardware. Preserve logs, raw streams, numeric
summaries, and rendered evidence as CI artifacts.

## Measure and optimize

Once the application is correct, measure it on hardware:

- inference time and cycles;
- capture, preprocessing, inference, and post-processing time;
- NPU-delegated and CPU fallback operators;
- memory use and placement of the model, pools, tensors, and I/O buffers.

Apply one optimization at a time and rerun the same regression data before
keeping it. Typical improvements include moving buffers between memory regions,
selecting a more appropriate Vela memory mode, folding preprocessing or a CPU
operator into the graph, and reducing the model or its working set.

## Common integration failures

Typical failures include:

- an NPU or Vela configuration that does not match the device or FVP;
- buffers placed outside memory accessible by both Cortex-M and Ethos-U;
- missing alignment, MPU attributes, or cache maintenance;
- incorrect Ethos-U interrupt routing;
- insufficient tensor, method-pool, or temporary-pool memory;
- application tensors whose shape or type differs from the exported program.

Diagnose the running target and inspect its build artefacts instead of relying
only on added logging. A faulting address, linker placement, NPU configuration,
and command-stream location usually narrow these problems quickly.

## What to keep and what to replace

| Keep from the template | Adapt for the application |
| --- | --- |
| CMSIS-Toolbox, AI-layer generation, and build flow | Board, device, debugger, and run configuration |
| ExecuTorch runtime and selected backend | PyTorch model and representative calibration data |
| `create_ai_layer.py` and generated layer structure | Input pipeline and output interpretation |
| Ethos-U integration pattern | Memory regions, cache policy, interrupts, and NPU configuration |
| CI structure and reference FVP targets | Hardware tests and captured real-world data |

The model is one component of the application. Completion means that the whole
system works with the target's actual memory, peripherals, timing, and data—and
that the evidence from each run feeds the next development iteration.
