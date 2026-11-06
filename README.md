# ExecuTorch on Cortex-M and Ethos-U

[![Build and test CMSIS-M](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-m.yml/badge.svg)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-m.yml)
[![Build and test CMSIS-U85](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-u85.yml/badge.svg)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-u85.yml)
[![Build and test CMSIS-U55](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-u55.yml/badge.svg)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-u55.yml)
[![Build and test Zephyr-U55](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-zephyr-u55.yml/badge.svg)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-zephyr-u55.yml)
[![Test Python environment](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-python-environment.yml/badge.svg)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-python-environment.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![CMSIS-Toolbox: 2.15](https://img.shields.io/badge/CMSIS--Toolbox-2.15-blue.svg)](https://open-cmsis-pack.github.io/cmsis-toolbox/)
[![ExecuTorch: 1.5.1](https://img.shields.io/badge/ExecuTorch-1.5.1-blue.svg)](https://www.keil.arm.com/packs/executorch-pytorch/)

This repository provides template applications for deploying and running
[ExecuTorch](https://github.com/pytorch/executorch) programs for ML models on
Cortex-M systems. The templates cover CMSIS and Zephyr software workflows, with
Cortex-M-only execution or acceleration by an Arm Ethos-U NPU.

Each template is a starting point for developing an embedded ML application
that runs one or more ExecuTorch programs on a Cortex-M platform. Adapt it to
specific hardware by supplying the device, memory, and execution configuration
through the selected solution and its CMSIS or Zephyr platform configuration.

- The [`PyTorch::ExecuTorch`](https://www.keil.arm.com/packs/executorch-pytorch/)
  pack provides the ExecuTorch runtime, required operators, and Ethos-U backend.
- In the CMSIS templates, the
  [`ARM::CMSIS-Ethos-U`](https://www.keil.arm.com/packs/cmsis-ethos-u-arm/overview/)
  pack provides the Ethos-U driver and reference-platform integration.
- The Zephyr templates use the Ethos-U driver included in upstream Zephyr 4.5
  and Zephyr's `hal_ethos_u` module.
- The build process uses [CMSIS-Toolbox 2.15.0](https://open-cmsis-pack.github.io/cmsis-toolbox/)
  or higher.

Every template uses the
[CMSIS-Toolbox MLOps flow](https://open-cmsis-pack.github.io/cmsis-toolbox/build-overview/#mlops-information)
to connect the selected embedded target with the model-conversion tools. The
generated `*.cbuild-mlops.yml` describes the processor, optional NPU, model
inputs, conversion settings, and build/run configuration. This allows the
resulting application and ML models to run and be validated on target hardware
or an FVP simulation model, following the development loop described in
[Making Edge AI development more productive](https://developer.arm.com/community/arm-community-blogs/b/tools-software-ides-blog/posts/making-edge-ai-development-more-productive).

For the complete path from selecting a template through FVP validation,
hardware deployment, repeatable real-world data tests, and optimization, see
[Create ExecuTorch applications for real devices](documentation/executorch-on-real-devices.md).

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

## Start a template

1. Create a repository from this GitHub template, clone it, and open its root
   directory in VS Code. For a Zephyr template, first prepare its Zephyr
   workspace as described in the template README, then open VS Code from a
   shell where the Zephyr Python environment is active.
2. In the CMSIS view, select **... > Select Active Solution from Workspace**,
   then select the `*.csolution.yml` file in one of the template directories
   below.
3. Select **Terminal > Run Task > Create Python environment (uv)**. The pip
   variant can be used when a compatible Python `>=3.10,<3.14` is installed.
4. Follow the selected directory's README to choose a target-set, generate the
   AI layer when needed, and build, run, or debug the application.

## Template workflows

Select a template directory and follow its README. Template descriptions come
from the corresponding `*.csolution.yml` file.

| Directory | Scope | Contents |
| --- | --- | --- |
| [`cmsis-u55/`](cmsis-u55/README.md) | CMSIS Cortex-M/U55/U65 | ExecuTorch template for Ethos-U55/U65 on Corstone-300 (SSE-300) FVP |
| [`cmsis-u85/`](cmsis-u85/README.md) | CMSIS Cortex-M/U85 | ExecuTorch template for Ethos-U85 on Corstone-320 (SSE-320) FVP |
| [`cmsis-m/`](cmsis-m/README.md) | CMSIS Cortex-M | ExecuTorch template for Cortex-M55 on Corstone-300 (SSE-300) FVP |
| [`zephyr-u55/`](zephyr-u55/README.md) | Zephyr Cortex-M/U55/U65 | Zephyr ExecuTorch template for Ethos-U55/U65 on Corstone-300 (SSE-300) FVP |
| [`zephyr-u85/`](zephyr-u85/README.md) | Zephyr Cortex-M/U85 | Zephyr ExecuTorch template for Ethos-U85 on Corstone-320 (SSE-320) FVP |
| [`zephyr-m/`](zephyr-m/README.md) | Zephyr Cortex-M | Zephyr ExecuTorch template for Cortex-M55 on the Alif Ensemble E8 AppKit and Corstone-300 FVP |
| [`model_conversion/`](model_conversion/README.md) | Shared model-conversion tools | Converts ML models into ExecuTorch programs and generates the AI layer and integration metadata for CMSIS or Zephyr builds |
| [`model/`](model/) | Shared model sources | One subdirectory per ML model |

## What the templates provide

- Converting one or more ML models into separate ExecuTorch programs and
  packaging them in one AI layer for a single Cortex-M processor, optionally
  with Ethos-U acceleration, using a shared Python virtual environment.
- Keeping applications small by linking only the ExecuTorch runtime, backends,
  and operators required by the programs in the AI layer.
- Simplifying model conversion by taking the processor and optional NPU/Vela
  configuration directly from the selected CMSIS or Zephyr solution.
- A three-step flow with a clean hand-over from the CMSIS-Toolbox to an MLOps system: the toolbox describes the target in a `*.cbuild-mlops.yml` file, a script turns that into the AI layer, and the toolbox builds the application.
- Running and validating the finished application on hardware or an FVP simulation model.

## How ML model generation works

The `*.csolution.yml` file describes the target and ML model inputs:

```yaml
mlops:
  npu:
    type: Ethos-U85
  vela:
    system: Ethos_U85_SYS_DRAM_Mid
    memory: Shared_Sram
  model:
    clayer: $AI-Layer$
    input:
      - name: TinyCNN
        file: ../model/TinyCNN/model.py
        parameters:
          input-shape: 1x16x16x3
          calibration-samples: 2
#     - name: model-b
#       ...
```

A three-step workflow generates the application:

```console
cbuild setup <solution>.csolution.yml --active <target-set> --packs      # create *.cbuild-mlops.yml
python model_conversion/create_ai_layer.py <solution>.cbuild-mlops.yml   # create ExecuTorch programs
cbuild <solution>.csolution.yml --active <target-set>                    # build application
```

By default, `create_ai_layer.py` converts every model declared in the
solution. Use `--model <name>` to update one model; the script then regenerates
the layer-wide metadata for all models.

See [the MLOps flow](documentation/mlops-flow.md) for a detailed walkthrough.

## Adapting the template

Add one subdirectory under `model/` for each ML model and declare every model
under `mlops.model.input`. The converter creates a separate ExecuTorch program
for each entry and packages all programs into one AI layer. Adapt the CMSIS
runner in `src/app_main.cpp` or the Zephyr runner in `src/executorch_test.cpp`
to load the required program and provide its input and output handling.

For an Ethos-U target, keep the NPU type, MAC count, Vela memory settings,
driver configuration, memory layout, and FVP or hardware description
consistent. CMSIS templates express platform settings through board layers;
Zephyr templates use Kconfig and devicetree. The
[CMSIS-Ethos-U integration guide](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html)
covers both CMSIS and Zephyr integration. After changing the target or model
list, rerun all three steps above.

When updating ExecuTorch, update the pack and Python package versions
together. More information is available in
[`PyTorch::ExecuTorch` pack maintenance and versioning](documentation/executorch-pack-maintenance.md).

## Generated AI layer

One AI layer contains all ML models that run on one CPU, optionally accelerated
by an NPU. `create_ai_layer.py` creates separate program artifacts for every ML
model and one shared build description for the complete AI layer:

```text
ai_layer/
|-- ai_layer.clayer.yml              # shared build description
|-- <model-a>.pte                    # serialized ExecuTorch program
|-- <model-a>_pte.c                  # embedded program data
|-- <model-a>_pte.h                  # program data declarations
|-- <model-a>_REPORT.md              # conversion and optimization summary
|-- <model-b>.pte
|-- <model-b>_pte.c
|-- <model-b>_pte.h
|-- <model-b>_REPORT.md
|-- CMakeLists.txt                   # Zephyr, when requested
|-- components.conf                  # Zephyr, when requested
`-- operators.conf                   # Zephyr, when requested
```

`ai_layer.clayer.yml` records the shared ExecuTorch runtime, required backends,
and combined CPU fallback operators for all programs. The optional Zephyr files
are generated from the same layer data.

## Continuous integration

Dedicated GitHub Actions workflows validate the three CMSIS templates and the
Zephyr-U55 template on every push and pull request to `main`. The CMSIS
workflows build with AC6, GCC, and Clang; the Zephyr workflow uses GCC. Each
workflow generates the target-specific AI layer, builds the application, runs
it on the matching FVP, checks `Test_result: PASS`, and uploads its artifacts.
A separate workflow verifies the Python environment setup on Linux, macOS, and
Windows.

## Project layout

| Path | Purpose |
| ------ | --------- |
| `cmsis-u85/` | Corstone-320 / Ethos-U85 solution, project, sources, layers, RTE configuration, and board support |
| `cmsis-u55/` | Corstone-300 / Ethos-U55/U65 solution, project, sources, layers, RTE configuration, and board support |
| `cmsis-m/` | Corstone-300 / Cortex-M55 CPU-only solution using the shared runner and board layer |
| `zephyr-u85/` | Zephyr 4.5 application for Alif Ensemble E8 hardware and the Corstone-320 FVP |
| `zephyr-u55/` | Zephyr 4.5 Ethos-U55/U65 application; provided targets use the Alif Ensemble E8 AppKit's Ethos-U55 and the Corstone-300 FVP |
| `zephyr-m/` | Zephyr 4.5 Cortex-M55 CPU-only application for the Alif Ensemble E8 AppKit using Zephyr's CMSIS-NN module |
| `model/TinyCNN/model.py` | Starter TinyCNN ML model; add sibling directories for more models |
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
| `documentation/executorch-pack-maintenance.md` | Where the `PyTorch::ExecuTorch` pack comes from, how to update it |

## License

The template code is licensed under Apache-2.0; see `LICENSE`. ExecuTorch and
the `arm_embedded_module.*` files under `cmsis-u*/src/` and `zephyr-u*/src/`,
which are derived from it, use a BSD-3-Clause license; see the adjacent
`LICENSE-ExecuTorch` file.

## References

- [`PyTorch::ExecuTorch` pack](https://www.keil.arm.com/packs/executorch-pytorch/)
- [ExecuTorch](https://github.com/pytorch/executorch)
- [ExecuTorch Arm Ethos-U backend](https://docs.pytorch.org/executorch/main/backends-arm-ethos-u.html)
- [CMSIS-Ethos-U documentation](https://arm-software.github.io/CMSIS-Ethos-U/)
- [CMSIS-Ethos-U integration guide](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html)
- [Configure 256 MACs for FVP simulation](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html#configure-256-macs-for-fvp-simulation)
- [CMSIS-Ethos-U pack](https://www.keil.arm.com/packs/cmsis-ethos-u-arm/overview/)
- [CMSIS-Toolbox MLOps information](https://open-cmsis-pack.github.io/cmsis-toolbox/build-overview/#mlops-information)
- [Arm CMSIS documentation](https://arm-software.github.io/CMSIS_6/latest/index.html)
