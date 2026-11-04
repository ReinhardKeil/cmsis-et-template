# ExecuTorch on Cortex-M and Ethos-U

[![Build and test CMSIS-U85](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-u85.yml/badge.svg)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-u85.yml)
[![Build and test CMSIS-U55](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-u55.yml/badge.svg)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-u55.yml)
[![Build and test Zephyr-U55](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-zephyr-u55.yml/badge.svg)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-zephyr-u55.yml)
[![Test Python environment](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-python-environment.yml/badge.svg)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-python-environment.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![CMSIS-Toolbox: 2.15](https://img.shields.io/badge/CMSIS--Toolbox-2.15-blue.svg)](https://open-cmsis-pack.github.io/cmsis-toolbox/)
[![ExecuTorch: 1.5.1](https://img.shields.io/badge/ExecuTorch-1.5.1-blue.svg)](https://www.keil.arm.com/packs/executorch-pytorch/)

This repository provides six template applications for deploying and running
[ExecuTorch](https://github.com/pytorch/executorch) ML models on Cortex-M
systems. The templates cover CMSIS and Zephyr software workflows, with
Cortex-M-only execution or acceleration by an Arm Ethos-U NPU.

Each template is a starting point that can be adopted for a specific hardware
platform by supplying its device, memory, and execution configuration through
the CMSIS project and software packs.

- The [`PyTorch::ExecuTorch`](https://www.keil.arm.com/packs/executorch-pytorch/)
  pack provides the ExecuTorch runtime, required operators, and Ethos-U backend.
- In the CMSIS templates, the
  [`ARM::CMSIS-Ethos-U`](https://www.keil.arm.com/packs/cmsis-ethos-u-arm/overview/)
  pack provides the Ethos-U driver and reference-platform integration.
- The Zephyr templates do not use `ARM::CMSIS-Ethos-U`. They use the Ethos-U
  driver included in upstream Zephyr 4.5 and Zephyr's `hal_ethos_u` module.
- The build process uses [CMSIS-Toolbox 2.15.0](https://open-cmsis-pack.github.io/cmsis-toolbox/)
  or higher.

Every template uses the
[CMSIS-Toolbox MLOps flow](https://open-cmsis-pack.github.io/cmsis-toolbox/build-overview/#mlops-information)
to connect the selected embedded target with the model-conversion tools. The
generated `*.cbuild-mlops.yml` describes the processor, optional NPU, model,
conversion settings, and build/run configuration. This allows the resulting
application and ML model to run and be validated on target hardware or an FVP
simulation model, following the development loop described in
[Making Edge AI development more productive](https://developer.arm.com/community/arm-community-blogs/b/tools-software-ides-blog/posts/making-edge-ai-development-more-productive).

For the complete path from selecting a template through FVP validation,
hardware deployment, repeatable real-world data tests, and optimization, see
[Create ExecuTorch applications for real devices](documentation/executorch-on-real-devices.md).

## Template workflows

Each template lives in its own directory and contains its application,
target-specific configuration, generated AI layer, and Quick Start guide.
Model sources and model-conversion tools are shared at workspace level.

Each solution defines one or more CMSIS-Toolbox target-sets. Select the
target-set for setup and build with `--active <target-set>`; this repository
does not use the older context terminology for that selection.

| Directory | Software workflow | ML execution | Availability |
| --- | --- | --- | --- |
| [`cmsis-u55/`](cmsis-u55/README.md) | CMSIS | Cortex-M with Ethos-U55 or Ethos-U65 | Available |
| [`cmsis-u85/`](cmsis-u85/README.md) | CMSIS | Cortex-M with Ethos-U85 | Available |
| `cmsis-m/` | CMSIS | Cortex-M only | Planned |
| [`zephyr-u55/`](zephyr-u55/README.md) | Zephyr | Cortex-M with Ethos-U55 or Ethos-U65 | Available |
| [`zephyr-u85/`](zephyr-u85/README.md) | Zephyr | Cortex-M with Ethos-U85 | Available |
| `zephyr-m/` | Zephyr | Cortex-M only | Planned |

The U55 directories also cover Ethos-U65 targets. Their integration flows are
close enough that separate `cmsis-u65/` and `zephyr-u65/` templates would only
duplicate the same structure; the selected hardware configuration supplies the
specific NPU parameters.

## What the templates provide

- Exporting an ExecuTorch ML model for Cortex-M or Ethos-U from a shared Python virtual environment.
- The pack [`PyTorch::ExecuTorch`](https://www.keil.arm.com/packs/executorch-pytorch/) links only the required and operator components that the ML model needs.
- Managing the target and optional NPU/Vela configuration in the CMSIS project rather than duplicating it in the Python exporter.
- A three-step flow with a clean hand-over from the CMSIS-Toolbox to an MLOps system: the toolbox describes the target in a `*.cbuild-mlops.yml` file, a script turns that into the AI layer, and the toolbox builds the application.
- Running and validating the finished application on hardware or an FVP simulation model.

## Configuring an Ethos-U template

Use the
[CMSIS-Ethos-U integration guide](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html)
when adapting an Ethos-U template to hardware. The NPU type, MAC count, memory
regions, and Vela configuration must describe the same hardware in every part
of the flow: the solution's `mlops:` settings, the generated model, the driver
and board configuration, the linker and memory attributes, and the FVP or
physical target.

For an FVP, keep its `ethosu.num_macs` parameter consistent with the target
configuration. The integration guide includes a concrete
[256-MAC FVP configuration example](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html#configure-256-macs-for-fvp-simulation).
After changing the NPU or its MAC count, run `cbuild setup`, regenerate the AI
layer so that Vela recompiles the model for that accelerator, and rebuild the
application.

The CMSIS templates express the driver configuration through the
`ARM::CMSIS-Ethos-U` Board layer. Zephyr templates apply the same configuration
consistency to their Kconfig and devicetree settings, but use Zephyr's upstream
Ethos-U driver rather than the `ARM::CMSIS-Ethos-U` pack.

## Prerequisites

- [Git](https://git-scm.com/) to clone the repository created from this template.
- [Keil Studio for VS Code](https://marketplace.visualstudio.com/items?itemName=Arm.keil-studio-pack) from the VS Code marketplace.
- [Python extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-python.python).
- [Astral's uv](https://docs.astral.sh/uv/getting-started/installation/) on `PATH` for the
  recommended environment task, or an installed Python `>=3.10,<3.14` for pip.
- On macOS, follow [FVPs on macOS](https://github.com/Arm-Examples/FVPs-on-Mac)
  to install Docker Desktop and configure the FVP environment.

Keil Studio manages the required license; the free Keil MDK Community edition
can be used for evaluation.

> [!NOTE]
> On Windows, enable [long-path support](https://learn.microsoft.com/windows/win32/fileio/maximum-file-path-limitation#enable-long-paths-in-windows-10-version-1607-and-later).

## Quick starts

Start with the guide in the selected template directory. The currently
available guides are:

- [CMSIS Cortex-M with Ethos-U55](cmsis-u55/README.md) — Corstone-300 / Ethos-U55-128.
- [CMSIS Cortex-M with Ethos-U85](cmsis-u85/README.md) — Corstone-320 / Ethos-U85-256.
- [Zephyr Cortex-M with Ethos-U85](zephyr-u85/README.md) — Alif Ensemble E8 hardware or Corstone-320 FVP.
- [Zephyr Cortex-M with Ethos-U55](zephyr-u55/README.md) — Alif Ensemble E8 AppKit or Corstone-300 FVP.

## How ML model generation works

The target is described by the `mlops:` node in
`cmsis-u85/cmsis-u85.csolution.yml`:

```yaml
mlops:
  npu:
    type: Ethos-U85
  vela:
    system: Ethos_U85_SYS_DRAM_Mid
    memory: Shared_Sram
  model:
    clayer: $AI-Layer$
    name: TinyCNN
    input-shape: 1x16x16x3
    calibration-samples: 2
```

`cbuild setup --active SSE-320-U85` enables the `SSE-320-U85` target-set and
resolves it into
`cmsis-u85/cmsis-u85.cbuild-mlops.yml`, which contains the processor, NPU
and Vela options, and the FVP simulation target-set as the `simulator:` to test
on.
`model_conversion/create_ai_layer.py` reads those options and passes all of them to ExecuTorch's
`EthosUCompileSpec`, so the target configuration is never duplicated in
Python. The two extra keys under `model:` are parameters of the ML model itself;
the toolbox passes them through and the script hands them to `model/model.py`.
The script then writes:

- `<solution>/ai_layer/ai_layer.clayer.yml`: the CMSIS components required by the ML model.
- `<solution>/ai_layer/model_pte.c` and `model_pte.h`: the ExecuTorch program embedded as a C array.
- `<solution>/ai_layer/model.pte`: the program itself, for inspection.

See [the MLOps flow](documentation/mlops-flow.md) for a detailed walkthrough.

## Component selection

The [ExecuTorch pack](https://www.keil.arm.com/packs/executorch-pytorch/)
provides the runtime, backends, and individual operators as selectable CMSIS
components. `model_conversion/create_ai_layer.py` examines the exported program and writes
`<solution>/ai_layer/ai_layer.clayer.yml` so only the required components are linked.
Because the layer is complete before the build starts, an ML model change that
changes the operator set needs nothing more than re-running steps 2 and 3.

## Adapting the template

To use a different ML model, replace or modify `model/model.py` and update the
ML model name or input handling as required: `get_model(input_shape)` builds the
ML model for the `input-shape` of the `mlops:` node, and
`get_calibration_inputs(input_shape, calibration_samples)` returns the samples
the quantizer is calibrated with; give it representative data for a trained
ML model. The runner in each solution's `src/app_main.cpp` builds its input tensor with the same
fixed shape and prints the output as ten floats, so an ML model with another
input shape or output needs matching changes there. Then re-run
`model_conversion/create_ai_layer.py` and build.

Give a vision ML model its input as NHWC, the layout of a camera frame, as
`model/model.py` does: the ML model permutes it to the NCHW its convolutions take,
and Vela folds that into the first convolution. With an NCHW input the NPU
first reorders the image in an operation of its own.

To target another Ethos-U configuration, update the target and `mlops:`
settings in the CMSIS solution and re-run all three steps. The generated Vela
options then follow that configuration automatically; `vela: misc:` takes any
further Vela option, for example `--optimise Size`. Moving to a different
board or reference platform also requires the corresponding device pack, board
support, memory layout, and FVP configuration. Follow the
[CMSIS-Ethos-U integration guide](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html)
and, when changing the simulated MAC count, its
[FVP configuration example](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html#configure-256-macs-for-fvp-simulation).

When updating ExecuTorch, update the pack and Python package versions
together. More information is available in
[ExecuTorch pack maintenance and versioning](documentation/executorch-pack-maintenance.md).

## Continuous integration

Dedicated GitHub Actions workflows validate each available template on every
push and pull request to `main`. The CMSIS workflows build with AC6, GCC, and
Clang; the Zephyr workflows use GCC. Each workflow generates the target-specific
AI layer, builds the application, runs it on the matching FVP, checks
`Test_result: PASS`, and uploads its artifacts. A separate workflow verifies
the Python environment setup on Linux, macOS, and Windows.

## Project layout

| Path | Purpose |
| ------ | --------- |
| `cmsis-u85/` | Corstone-320 / Ethos-U85 solution, project, sources, layers, RTE configuration, and board support |
| `cmsis-u55/` | Corstone-300 / Ethos-U55-128 solution, project, sources, layers, RTE configuration, and board support |
| `zephyr-u85/` | Zephyr 4.5 application for Alif Ensemble E8 hardware and the Corstone-320 FVP |
| `zephyr-u55/` | Zephyr 4.5 application for the Alif Ensemble E8 AppKit's Ethos-U55 and the Corstone-300 FVP |
| `model/model.py` | Starter TinyCNN ML model |
| `model_conversion/create_ai_layer.py` | Exports the ML model for the target and writes the AI layer |
| `model_conversion/README.md`, `model_conversion/model_conversion-specification.md` | Usage and specification for the model-conversion toolset |
| `model_conversion/test/` | Unit tests for the AI-layer generator |
| `model_conversion/setup_venv.py` (`.sh` / `.bat`) | Creates the Python environment for the export |
| `.vscode.d/tasks.json` | The VS Code tasks (venv setup, Create AI layer) merged by the CMSIS Solution extension |
| `.vscode/fvp.sh`, `.vscode/fvp.Dockerfile`, `.vscode/launch.json.mac` | macOS only: runs the FVP in Docker, and the debug configuration that waits for it |
| `cmsis-u*/src/app_main.cpp` | Loads the ML model, runs inference, and prints the result; pool sizes overridable with `APP_METHOD_POOL_SIZE`, `APP_TEMP_POOL_SIZE`, `APP_POOL_SECTION` |
| `cmsis-u*/src/arm_embedded_module.*` | `EmbeddedModule`: ExecuTorch's `Module` class without the POSIX file loading (BSD-3-Clause, `src/LICENSE-ExecuTorch`) |
| `documentation/mlops-flow.md` | The MLOps flow in detail |
| `documentation/executorch-on-real-devices.md` | Adapting a template, validating it on an FVP, and deploying it to hardware |
| `documentation/executorch-pack-maintenance.md` | Where the ExecuTorch pack comes from, how to update it |

## Usage notes

Each solution's generated `ai_layer/` is target-specific. After switching targets, run
`model_conversion/create_ai_layer.py` with the selected solution's
`*.cbuild-mlops.yml`. In VS Code, the task obtains its path from
`${command:cmsis-csolution.getSolutionDir}`; `.venv` remains under
`${workspaceFolder}`.

## License

The template code is licensed under Apache-2.0; see `LICENSE`. ExecuTorch and
the `arm_embedded_module.*` files under `cmsis-u*/src/` and `zephyr-u*/src/`,
which are derived from it, use a BSD-3-Clause license; see the adjacent
`LICENSE-ExecuTorch` file.

## References

- [PyTorch ExecuTorch pack](https://www.keil.arm.com/packs/executorch-pytorch/)
- [ExecuTorch](https://github.com/pytorch/executorch)
- [ExecuTorch Arm Ethos-U backend](https://docs.pytorch.org/executorch/main/backends-arm-ethos-u.html)
- [CMSIS-Ethos-U documentation](https://arm-software.github.io/CMSIS-Ethos-U/)
- [CMSIS-Ethos-U integration guide](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html)
- [Configure 256 MACs for FVP simulation](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html#configure-256-macs-for-fvp-simulation)
- [CMSIS-Ethos-U pack](https://www.keil.arm.com/packs/cmsis-ethos-u-arm/overview/)
- [CMSIS-Toolbox MLOps information](https://open-cmsis-pack.github.io/cmsis-toolbox/build-overview/#mlops-information)
- [Arm CMSIS documentation](https://arm-software.github.io/CMSIS_6/latest/index.html)
