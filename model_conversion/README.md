# ExecuTorch model conversion

This directory contains the complete host-side toolset for converting the
PyTorch model into an ExecuTorch program for an Arm Cortex-M or Ethos-U target and
generating the corresponding CMSIS AI layer.

The converter reads the target and output information produced by
CMSIS-Toolbox in a `*.cbuild-mlops.yml` file. It quantizes `model/model.py`,
uses either the Cortex-M backend with CMSIS-NN or the Ethos-U backend with
Vela, and generates the files needed by the CMSIS build.

[CMSIS-Toolbox MLOps integration](https://open-cmsis-pack.github.io/cmsis-toolbox/build-overview/#mlops-integration)
describes how the input file is generated.

[Model-conversion toolset specification](model_conversion-specification.md)
defines every deliverable in this directory and its behavioral contract.

The workspace and solution directories have distinct roles:

- The **workspace directory** contains `model_conversion/` and `.venv/`.
  One workspace environment is shared by all solutions.
- A **solution directory** contains a `.csolution.yml`, its generated
  `*.cbuild-mlops.yml`, `model/model.py`, Vela configuration, and generated AI
  layer. A workspace may contain multiple solution directories.

## Set up the environment

Run the appropriate wrapper from the workspace directory.

Linux and macOS:

```console
./model_conversion/setup_venv.sh
```

Windows:

```console
.\model_conversion\setup_venv.bat
```

The setup creates `.venv` in the workspace directory. It installs
`requirements.txt` normally, then installs `requirements-arm-tosa.txt` with
`--no-deps` to preserve the FlatBuffers version required by ExecuTorch.

Use `--recreate` to rebuild an existing environment. The optional `--uv` and
`--python VERSION` arguments select uv and a supported Python version:

```console
./model_conversion/setup_venv.sh --uv --python 3.12 --recreate
```

## Convert a model

Pass the MLOps file belonging to the selected solution. For a solution in the
workspace root:

```console
cbuild setup cmsis-u85/cmsis-u85.csolution.yml --active SSE-320-U85 --packs
python model_conversion/create_ai_layer.py cmsis-u85/cmsis-u85.cbuild-mlops.yml
cbuild cmsis-u85/cmsis-u85.csolution.yml --active SSE-320-U85
```

The `--active SSE-320-U85` option enables that CMSIS-Toolbox target-set for
both setup and build.

`create_ai_layer.py` automatically restarts itself with the `.venv` interpreter
when necessary. It resolves `.venv` from the workspace directory. The
directory containing the supplied `*.cbuild-mlops.yml` is the solution
directory, regardless of the shell's current working directory. It resolves
`model/model.py`, Vela configuration, pack metadata, and `model.clayer`
relative to that solution directory.

For a solution in a workspace subdirectory:

```console
python model_conversion/create_ai_layer.py solutions/demo/demo.cbuild-mlops.yml
```

## Input

For an Ethos-U target, the `*.cbuild-mlops.yml` file provides the `npu:` and
`vela:` configuration described in the
[CMSIS-Toolbox MLOps documentation](https://open-cmsis-pack.github.io/cmsis-toolbox/YML-Input-Format/#mlops-management).
For a CPU-only target, omit `npu:`. If the selected device pack contributes NPU
information even though the application does not use it, set `model.cpu` to
the processor name defined by Open-CMSIS-Pack, for example `Cortex-M55`. The
generator then derives its `CortexMTargetConfig` from `model.cpu` (or otherwise
from `processor.type`) and uses the Cortex-M quantizer and lowering passes. The
file also provides the following `model:` settings used by
`create_ai_layer.py`. Add these settings under `mlops.model` in the
`*.csolution.yml`; `cbuild setup` copies them to the `*.cbuild-mlops.yml` file.

### `model:`

| `model:` | | Content |
|---|---|---|
| `clayer:` | Required | Path and filename of the `*.clayer.yml` file used to integrate the model into a CMSIS-style project. Its parent folder also receives the generated model files. |
| `zephyr:` | Optional | Enables generation of additional Zephyr integration files in the clayer path. A plain empty node and `zephyr: {}` are equivalent. |
| `cpu:` | Optional | Open-CMSIS-Pack processor name, such as `Cortex-M55`; selects Cortex-M generation even when the device pack contributes NPU information. |
| `name:` | Optional | Model name used for the CMSIS layer group and as the layer-description fallback. |
| `input-shape:` | Optional | Model input dimensions separated by `x` or commas, for example `1x16x16x3`. Passed to `get_model` and `get_calibration_inputs` as `input_shape`. |
| `calibration-samples:` | Optional | Number of calibration inputs. Converted to an integer and passed to `get_calibration_inputs`. |

The matching `*.cbuild-pack.yml` file supplies the selected `PyTorch::ExecuTorch` pack
version.

## Output

The script uses the parent folder of the `*.clayer.yml` path specified by
`model.clayer` as its output folder. For example, with
`clayer: ai_layer/ai_layer.clayer.yml`, generated files are written to
`ai_layer/`.

The script creates:

- `model_pte.c` and `model_pte.h`, which embed the program as a C array; and
- `model.pte` for inspection; and
- `VELA_SUMMARY.md` for Ethos-U generation, containing the model name,
  effective accelerator, system configuration, memory mode, and complete Vela
  textual report.

The script also creates the `*.clayer.yml` file specified by `model.clayer`.
This file selects the required ExecuTorch runtime, target backend, and operator
components. CPU-only layers omit `Backend EthosU` and add the unversioned
`ARM::CMSIS-NN` pack plus `CMSIS:NN Lib` when selected operators require it.

When the `zephyr:` node is present, the script additionally creates
`CMakeLists.txt`, `components.conf`, and `operators.conf` for the Zephyr
application. CPU-only fragments omit the Ethos-U backend. No separate
`west ailayer` step is required.

The application root must append the two generated `.conf` files to
`EXTRA_CONF_FILE` before `find_package(Zephyr)`, then call
`add_subdirectory(ai_layer)` afterward. This lets the generated layer attach
`model_pte.c` directly and exactly once to Zephyr's `app` target.

All output files use LF line endings where applicable. The script stages the
complete generated layer in temporary storage and atomically replaces the
destination files only after export and output generation succeed. If Vela or
later generation fails, the previous AI layer remains intact. The generated
`*.cbuild-mlops.yml` input is never modified.

## Tests

Run the unit tests from the repository root:

```console
python -m unittest discover -s model_conversion/test -p "test_*.py" -v
```

The tests mock model export and do not require Torch, Vela, an FVP, or target
hardware.
