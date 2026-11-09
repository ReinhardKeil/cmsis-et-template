# ExecuTorch on Cortex-M and Ethos-U

[![Build and test CMSIS-M](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-m.yml/badge.svg)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-m.yml)
[![Build and test CMSIS-U85](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-u85.yml/badge.svg)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-u85.yml)
[![Build and test CMSIS-U55](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-u55.yml/badge.svg)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-u55.yml)
[![Build and test Zephyr-U55](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-zephyr-u55.yml/badge.svg)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-zephyr-u55.yml)
[![Test Python environment](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-python-environment.yml/badge.svg)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-python-environment.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![CMSIS-Toolbox: 2.15](https://img.shields.io/badge/CMSIS--Toolbox-2.15-blue.svg)](https://open-cmsis-pack.github.io/cmsis-toolbox/)
[![ExecuTorch: 1.5.1](https://img.shields.io/badge/ExecuTorch-1.5.1-blue.svg)](https://www.keil.arm.com/packs/executorch-pytorch/)

This repository provides template applications for deploying ML models on
Cortex-M targets using [ExecuTorch](https://github.com/pytorch/executorch).
The templates support CMSIS and Zephyr applications on Cortex-M-only systems,
where inference runs entirely on the CPU, and Cortex-M systems with an Arm
Ethos-U NPU, where supported model operations run on the NPU and the remaining
operations run on the CPU.

The templates support multiple ML models in one application and link only the
runtime, backends, and operators those models need.

## Software components

- The [`PyTorch::ExecuTorch`](https://www.keil.arm.com/packs/executorch-pytorch/)
  pack provides the ExecuTorch runtime, required operators, and Ethos-U backend.
- In the CMSIS templates, the
  [`ARM::CMSIS-Ethos-U`](https://www.keil.arm.com/packs/cmsis-ethos-u-arm/overview/)
  pack provides the Ethos-U driver and reference-platform integration.
- The Zephyr templates use the Ethos-U driver included in upstream Zephyr 4.5
  and Zephyr's `hal_ethos_u` module.
- [CMSIS-Toolbox 2.15.0](https://open-cmsis-pack.github.io/cmsis-toolbox/) or
  higher manages the build and passes target settings to the model converter.

## Prerequisites

- [Git](https://git-scm.com/).
- [Keil Studio for VS Code](https://marketplace.visualstudio.com/items?itemName=Arm.keil-studio-pack) from the VS Code marketplace.
- [Python extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-python.python).
- [Astral's uv](https://docs.astral.sh/uv/getting-started/installation/) on `PATH` for the
  Python environment task, or Python `>=3.10,<3.14` for the pip alternative.
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

## Choose a template

Choose a template for your software stack and processor/NPU combination.
Each template README lists the supplied board and FVP configurations and
explains how to use them. Adapt the template to your target hardware.

| Template | Target |
| --- | --- |
| [`cmsis-u55/`](cmsis-u55/README.md) | Cortex-M with Ethos-U55/U65 |
| [`cmsis-u85/`](cmsis-u85/README.md) | Cortex-M with Ethos-U85 |
| [`cmsis-m/`](cmsis-m/README.md) | Cortex-M CPU only |
| [`zephyr-u55/`](zephyr-u55/README.md) | Cortex-M with Ethos-U55/U65 |
| [`zephyr-u85/`](zephyr-u85/README.md) | Cortex-M with Ethos-U85 |
| [`zephyr-m/`](zephyr-m/README.md) | Cortex-M CPU only |

## Convert models and build

Define your target and model inputs in the `*.csolution.yml` file:

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

The [CMSIS-Toolbox MLOps integration](https://open-cmsis-pack.github.io/cmsis-toolbox/build-overview/#mlops-information)
has three steps. First, the toolbox writes the target and conversion settings
to `*.cbuild-mlops.yml`. Next, the converter creates the AI layer: the model
programs and their shared build configuration. Finally, the toolbox builds
the application.

```console
cbuild setup <solution>.csolution.yml --active <target-set> --packs      # create *.cbuild-mlops.yml
python model_conversion/create_ai_layer.py <solution>.cbuild-mlops.yml   # create ExecuTorch programs
cbuild <solution>.csolution.yml --active <target-set>                    # build application
```

By default, `create_ai_layer.py` processes every input declared in the
solution. Use `--input <name>` to update one input; the script then regenerates
the layer-wide metadata from all current validated output programs.

See [the MLOps integration](documentation/mlops-flow.md) for a detailed walkthrough
and [model-conversion tools](model_conversion/README.md) for converter options.

## Adapting the template

Add a subdirectory under `model/` for each model and list it under
`mlops.model.input`. Adapt the CMSIS runner in `src/app_main.cpp` or the Zephyr
runner in `src/executorch_test.cpp` to load the model and handle its inputs and
outputs.

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

For guidance on FVP validation, hardware deployment, real-world data tests,
and optimization, see
[Create ExecuTorch applications for real devices](documentation/executorch-on-real-devices.md).

## Generated AI layer

`create_ai_layer.py` writes the following files to `ai_layer/`:

```text
ai_layer/
|-- ai_layer.clayer.yml              # shared build description
|-- <input-a>.pte                    # backend-lowered program validated for target
|-- <input-a>_pte.c                  # embedded program data
|-- <input-a>_pte.h                  # program data declarations
|-- <input-a>_REPORT.md              # processing, lowering, and validation summary
|-- <input-b>.pte
|-- <input-b>_pte.c
|-- <input-b>_pte.h
|-- <input-b>_REPORT.md
|-- CMakeLists.txt                   # Zephyr: model sources and include directory
|-- components.conf                  # Zephyr: runtime and backend Kconfig settings
`-- operators.conf                   # Zephyr: required CPU operator Kconfig settings
```

`ai_layer.clayer.yml` records the shared ExecuTorch runtime, required backends,
and combined CPU fallback operators for all programs. When Zephyr output is
requested, the generator also creates:

- `CMakeLists.txt` to add the generated model C files and their header directory
  to the application build.
- `components.conf` to enable the ExecuTorch runtime, kernel utilities,
  registration code, and required backends through Kconfig.
- `operators.conf` to enable the CPU operators required by the models, including
  operations that are not delegated to an NPU.

## Continuous integration

Dedicated GitHub Actions workflows validate the three CMSIS templates and the
Zephyr-U55 template on every push and pull request to `main`. The CMSIS
workflows build with AC6, GCC, and Clang; the Zephyr workflow uses GCC. Each
workflow generates the target-specific AI layer, builds the application, runs
it on the matching FVP, checks `Test_result: PASS`, and uploads its artifacts.
A separate workflow verifies the Python environment setup on Linux, macOS, and
Windows.

## Shared files and tools

| Path | Purpose |
| ------ | --------- |
| `model/TinyCNN/model.py` | Starter TinyCNN ML model; add sibling directories for more models |
| `model_conversion/create_ai_layer.py` | Exports the ML model for the target and writes the AI layer |
| `model_conversion/README.md`, `model_conversion/model_conversion-specification.md` | Usage and specification for the model-conversion toolset |
| `model_conversion/test/` | Unit tests for the AI-layer generator |
| `model_conversion/setup_venv.py` (`.sh` / `.bat`) | Creates the Python environment for the export |
| `.vscode.d/tasks.json` | The VS Code tasks (venv setup, Create AI layer) merged by the CMSIS Solution extension |
| `.vscode/fvp.sh`, `.vscode/fvp.Dockerfile`, `.vscode/launch.json.mac` | macOS only: runs the FVP in Docker, and the debug configuration that waits for it |
| `cmsis-u*/src/app_main.cpp` | Loads the ML model, runs inference, and prints the result; pool sizes overridable with `APP_METHOD_POOL_SIZE`, `APP_TEMP_POOL_SIZE`, `APP_POOL_SECTION` |
| `cmsis-u*/src/arm_embedded_module.*` | `EmbeddedModule`: ExecuTorch's `Module` class without the POSIX file loading (BSD-3-Clause, `src/LICENSE-ExecuTorch`) |

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
