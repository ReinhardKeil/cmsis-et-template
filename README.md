# ExecuTorch Template for Cortex-M and Ethos-U

[![Build and test CMSIS-M](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-m.yml/badge.svg)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-m.yml)
[![Build and test CMSIS-U85](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-u85.yml/badge.svg)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-u85.yml)
[![Build and test CMSIS-U55](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-u55.yml/badge.svg)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-cmsis-u55.yml)
[![Build and test Zephyr (GCC)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-zephyr-gcc.yml/badge.svg)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-zephyr-gcc.yml)
[![Test CMSIS model variants](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/cmsis-variants.yml/badge.svg)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/cmsis-variants.yml)
[![Test Zephyr model variants](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/zephyr-variants.yml/badge.svg)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/zephyr-variants.yml)
[![Test Python environment](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-python-environment.yml/badge.svg)](https://github.com/reinhardkeil/cmsis-et-template/actions/workflows/test-python-environment.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![CMSIS-Toolbox: 2.16](https://img.shields.io/badge/CMSIS--Toolbox-2.16-blue.svg)](https://open-cmsis-pack.github.io/cmsis-toolbox/)
[![ExecuTorch: 1.6.0](https://img.shields.io/badge/ExecuTorch-1.6.0-blue.svg)](https://www.keil.arm.com/packs/executorch-pytorch/)

This repository provides ready-to-adapt templates for deploying
[ExecuTorch](https://github.com/pytorch/executorch) models on Arm Cortex-M
systems. Choose a CMSIS or Zephyr template for CPU-only inference or for
hardware-accelerated inference with an Arm Ethos-U NPU.

<img src="https://raw.githubusercontent.com/Arm-Examples/.github/main/profile/executorch-models.svg" alt="ExecuTorch model deployment flow" width="500">

## What you can do

Use these templates to:

- [Choose the CMSIS or Zephyr target](#choose-a-template) for Cortex-M CPU-only
  execution or hardware acceleration with an Arm Ethos-U NPU, then
  [start the template](#start-a-template).
- Verify the working Cortex-M and ExecuTorch integration with the included
  [TinyCNN example](model/TinyCNN/README.md).
- [Create a reusable AI layer](#create-the-ai-layer) from one or more ML models.
- [Adapt the template](#adapting-the-template) for the application-specific
  models, input and output processing, target, and application logic.
- [Validate the application](#validate-the-application) on an FVP and target
  hardware with repeatable inputs.
- [Use continuous integration](#continuous-integration) to test the CMSIS or Zephyr application with repeatable input data.

## Software components

- The [`PyTorch::ExecuTorch`](https://www.keil.arm.com/packs/executorch-pytorch/overview/)
  pack provides the ExecuTorch runtime, required operators, and Ethos-U backend.
- In the CMSIS templates, the
  [`ARM::CMSIS-Ethos-U`](https://www.keil.arm.com/packs/cmsis-ethos-u-arm/overview/)
  pack provides the Ethos-U driver and reference-platform integration.
- The Zephyr templates use the Ethos-U driver included in upstream Zephyr 4.5
  and Zephyr's `hal_ethos_u` module.
- [CMSIS-Toolbox 2.16.0](https://open-cmsis-pack.github.io/cmsis-toolbox/) or
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

## Choose a template

Choose a template for your software stack and processor/NPU combination.
Each template README lists the supplied board and FVP configurations and
explains how to use them. Adapt the template to your target hardware.

Each solution keeps its application sources, headers, compatibility support,
and board files inside its own directory. Build files must not reference
sources from another solution. CMSIS packs and Zephyr modules provide external
dependencies.

| Template | Target |
| --- | --- |
| [`cmsis-u55/`](cmsis-u55/README.md) | Cortex-M with Ethos-U55/U65 |
| [`cmsis-u85/`](cmsis-u85/README.md) | Cortex-M with Ethos-U85 |
| [`cmsis-m/`](cmsis-m/README.md) | Cortex-M CPU only |
| [`zephyr-u55/`](zephyr-u55/README.md) | Cortex-M with Ethos-U55/U65 |
| [`zephyr-u85/`](zephyr-u85/README.md) | Cortex-M with Ethos-U85 |
| [`zephyr-m/`](zephyr-m/README.md) | Cortex-M CPU only |

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

## Create the AI layer

[`create_ai_layer.py`](model_conversion/README.md) supports three input workflows:

| Input | Role | Action |
| --- | --- | --- |
| `.pt2` | Recommended lowering input | Validate, backend-lower, serialize as `.pte`, and integrate. |
| `.pte` | Optional prebuilt deployment artifact | Inspect, validate, and integrate unchanged. |
| `.py` | Convenience model-production input | Export, quantize, backend-lower, serialize, and integrate. |

A `.pt2` input contains one or more quantized `ExportedProgram`s. Model-specific
preparation, including calibration, quantization, and accuracy validation, takes
place before `create_ai_layer.py`; the tool applies target-specific lowering and
Vela processing for Ethos-U. All named programs in one `.pt2` become methods in
that input's output `.pte` container. A `.pte` must already be backend-lowered
and compatible with the selected target. The tool never runs Vela on `.pte`
inputs and rejects portable-only NPU inputs, incompatible programs, and
programs that cannot be verified.

One AI Layer targets exactly one CPU, optionally with Ethos-U acceleration,
but is independent of the application framework. The same layer can be used in
CMSIS and Zephyr projects for that target. For example, use CMSIS and the
[Synchronous Data Stream (SDS) Framework](https://github.com/ARM-software/SDS-Framework)
to validate the layer on hardware or an FVP, then deploy the unchanged layer in
either a CMSIS or Zephyr application.

The `.py` convenience workflow performs the complete process for simple models
and examples. Its Python workflow defines or loads the `torch.nn.Module` and
provides separate example and calibration inputs. Use `.pt2` when a model
requires custom quantization, QAT, specialized calibration, or model-specific
accuracy validation.

Define the target and inputs in the `*.csolution.yml` file. The file extension
selects the workflow:

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
        file: ../model/TinyCNN/TinyCNN.py
        parameters:
          input-shape: 1x16x16x3
          calibration-samples: 2
      - name: vision
        file: ../models/vision.pt2
      - name: diagnostics
        file: ../models/diagnostics.pte
```

The model requires `clayer:` for both CMSIS and Zephyr. For Zephyr, add an
empty `zephyr:` node alongside `clayer:` to generate the additional CMake and
Kconfig files. All output is written to the directory containing the clayer.
The Zephyr csolutions define `AI-Layer` in each target's variables and use
`clayer: $AI-Layer$`, matching the CMSIS templates.

Every input requires `name` and `file`. The optional `parameters` mapping is
valid only for `.py`. On a mixed CPU/NPU target, optional `cpu:` selects CPU
lowering for a `.py` or `.pt2` input. A `.pte` accepts no conversion or lowering
options because its backend and target compatibility are established by
inspection.

The [CMSIS-Toolbox MLOps integration](https://open-cmsis-pack.github.io/cmsis-toolbox/build-overview/#mlops-information)
has three steps. First, the toolbox writes the target and processing settings
to `*.cbuild-mlops.yml`. Next, `create_ai_layer.py` creates the output programs
and their shared build configuration. Finally, the toolbox builds the
application.

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

List each input artifact under `mlops.model.input`. For the `.py` convenience
workflow, add a subdirectory under `model/` for each model. Adapt the CMSIS
runner in `src/app_main.cpp` or the Zephyr runner in
`src/executorch_test.cpp` to load the model and handle its inputs and outputs.

For an Ethos-U target, keep the NPU type, MAC count, Vela memory settings,
driver configuration, memory layout, and FVP or hardware description
consistent. CMSIS templates express platform settings through board layers;
Zephyr templates use Kconfig and devicetree. The
[CMSIS-Ethos-U integration guide](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html)
covers both CMSIS and Zephyr integration. After changing the target or model
list, rerun all three steps above.

When updating ExecuTorch, follow
[Update the ExecuTorch version](model_conversion/README.md#update-the-executorch-version)
to update the Python environment, regenerate the AI layers, and update the
separately selected Zephyr external-module pack.

## Validate the application

Before adapting a template, run the [TinyCNN](model/TinyCNN/README.md) test on
the matching FVP and confirm `Test_result: PASS`. This known-good test separates
environment and tool problems from changes to the model or application.

When moving to hardware, first run the same generated `.pte` program, input,
and output comparison without enabling live sensors. The results should satisfy
the same comparison criteria on the FVP and hardware; fully integer execution
is normally expected to match exactly.
Follow the
[CMSIS-Ethos-U integration guide](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html)
when a difference points to Vela configuration, memory placement, cache
maintenance, interrupt routing, or driver configuration.

When adding an ML model, implement repeatable tests using captured or generated
input data. Replay the same inputs on the FVP and target hardware. Keep model
inputs, raw model outputs, and post-processed application results as separate
artifacts so regressions can be isolated.
The [Synchronous Data Stream Framework](https://arm-software.github.io/SDS-Framework/main/overview.html)
can record and replay audio, video, sensor, and intermediate-result streams on
hardware and FVPs.

Preserve the console log, generated model and Vela reports, captured inputs,
raw outputs, comparison tolerances, and any visual evidence as CI artifacts.
Compare floating-point and quantized results using ML model-specific
tolerances. Apply the same test criteria to the quantized program on the FVP
and hardware.

## Generated AI layer

The `mlops.model.clayer` setting selects the path and filename of the generated
`*.clayer.yml` file. `create_ai_layer.py` writes the remaining artifacts to its
parent directory:

```text
<clayer-directory>/
|-- <clayer-name>.clayer.yml         # layer description for CMSIS and other tools
|-- <name-a>.pte                     # backend-lowered program validated for target
|-- <name-a>_pte.c                   # embedded program data
|-- <name-a>_pte.h                   # program data declarations
|-- <name-a>_REPORT.md               # processing, lowering, and validation summary
|-- <name-b>.pte
|-- <name-b>_pte.c
|-- <name-b>_pte.h
|-- <name-b>_REPORT.md
|-- CMakeLists.txt                   # Zephyr: model sources and include directory
|-- components.conf                  # Zephyr: runtime and backend Kconfig settings
`-- operators.conf                   # Zephyr: required CPU operator Kconfig settings
```

The generated `*.clayer.yml` file records the shared ExecuTorch runtime,
required backends, and combined CPU fallback operators for all programs. When
`mlops.model.zephyr` is present, the generator also adds:

- `CMakeLists.txt` to add the generated model C files and their header directory
  to the application build.
- `components.conf` to enable the ExecuTorch runtime, kernel utilities,
  registration code, and required backends through Kconfig.
- `operators.conf` to enable the CPU operators required by the models, including
  operations that are not delegated to an NPU.

## Continuous integration

Dedicated GitHub Actions workflows validate all three CMSIS templates and all
three Zephyr templates on every push and pull request to `main`. The CMSIS
workflows build with AC6, GCC, and Clang; the Zephyr workflow uses GCC. The
CMSIS workflows regenerate the target-specific AI layer, while the Zephyr
workflow validates the committed AI layers. Each workflow builds the
application, runs it on the matching FVP, checks `Test_result: PASS`, and
uploads its artifacts. A separate workflow verifies the Python environment
setup on Linux, macOS, and Windows.

## Run MLOps separately from the embedded build

Model conversion may run on a separate host or CI system, independent of the
embedded application build. Generate the `*.cbuild-mlops.yml` of the solution with:

```console
cbuild setup <solution-dir>/<solution>.csolution.yml --active <target> --packs
```

### MLOps input

These are all files the embedded project needs to provide for the ML model conversion
step:

- The generated `*.cbuild-mlops.yml` file.
- The Vela INI files in the `.cmsis` folder referenced by that file.

Run the ML model conversion with:

```console
python model_conversion/create_ai_layer.py <solution-dir>/<solution>.cbuild-mlops.yml
```

This script may be adapted to the requirements of the MLOps system. For
details, see the [`model_conversion` README](model_conversion/README.md).

### MLOps output

The output is the content of the `ai_layer` folder referenced by `model.clayer`
in the `*.cbuild-mlops.yml` file. Transfer the complete folder to the embedded
build system.

## License

The template code is licensed under Apache-2.0; see `LICENSE`. ExecuTorch and
the `arm_embedded_module.*` files under each template's `src/` directory,
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
- [`ARM::SDS` pack](https://www.keil.arm.com/packs/sds-arm/overview/)
- [CMSIS-Toolbox MLOps information](https://open-cmsis-pack.github.io/cmsis-toolbox/build-overview/#mlops-information)
- [Arm CMSIS documentation](https://arm-software.github.io/CMSIS_6/latest/index.html)
