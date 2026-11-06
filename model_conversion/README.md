# ExecuTorch model conversion

This directory contains the complete host-side toolset for converting one or
more ML models into ExecuTorch programs for Arm Cortex-M, optionally with
Ethos-U acceleration, and generating one CMSIS AI layer for them.

The converter reads the target and output information produced by
CMSIS-Toolbox in a `*.cbuild-mlops.yml` file. Python model sources are
quantized and lowered with either the Cortex-M backend and CMSIS-NN or the
Ethos-U backend and Vela. Existing target-specific `.pte` files can also be
imported. Each model keeps separate artifacts; the AI layer contains their
shared runtime and combined CPU fallback-operator requirements.

[CMSIS-Toolbox MLOps integration](https://open-cmsis-pack.github.io/cmsis-toolbox/build-overview/#mlops-integration)
describes how the input file is generated.

[Model-conversion toolset specification](model_conversion-specification.md)
defines every deliverable in this directory and its behavioral contract.

The workspace and solution directories have distinct roles:

- The **workspace directory** contains `model_conversion/` and `.venv/`.
  One workspace environment is shared by all solutions.
- A **solution directory** contains a `.csolution.yml`, its generated
  `*.cbuild-mlops.yml`, model inputs, Vela configuration, and generated AI
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

`requirements.txt` pins a compatible ExecuTorch, PyTorch, TorchAO, and Vela
set. The ExecuTorch Python version must match the `PyTorch::ExecuTorch` pack
selected by the solution because that runtime consumes the generated `.pte`
files. `gen_oplist` and `gen_all_oplist` are included with ExecuTorch; TorchGen
is included with PyTorch.

Use `--recreate` to rebuild an existing environment. The optional `--uv` and
`--python VERSION` arguments select uv and a supported Python version:

```console
./model_conversion/setup_venv.sh --uv --python 3.12 --recreate
```

## Convert models

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
model inputs, Vela configuration, pack metadata, and `model.clayer` relative
to that solution directory.

For a solution in a workspace subdirectory:

```console
python model_conversion/create_ai_layer.py solutions/demo/demo.cbuild-mlops.yml
```

Without additional options, every declared model is converted. To update one
model while retaining the others, use:

```console
python model_conversion/create_ai_layer.py solutions/demo/demo.cbuild-mlops.yml --model TinyCNN
```

The selected model is converted again. The generator then inspects every
declared `.pte` file and rebuilds the layer-wide fallback-operator union and
metadata. A per-model report is output only; it is never operator-selection
input.

## Input

For an Ethos-U target, the `*.cbuild-mlops.yml` file provides the `npu:` and
`vela:` configuration described in the
[CMSIS-Toolbox MLOps documentation](https://open-cmsis-pack.github.io/cmsis-toolbox/YML-Input-Format/#mlops-management).
For a CPU-only target, omit `npu:`. On a mixed target, an empty `cpu:` key in a
model entry selects CPU-only conversion for that model. Accelerator, system
configuration, and memory mode are shared by every NPU model in the layer.

Declare models under `mlops.model.input` in the `*.csolution.yml` file.
CMSIS-Toolbox writes the generated form as `cbuild-mlops.model.input`:

```yaml
mlops:
  model:
    clayer: $AI-Layer$
    input:
      - name: TinyCNN
        file: ../model/TinyCNN/model.py
        optimize: size
        parameters:
          input-shape: 1x16x16x3
          calibration-samples: 2
      - name: CpuModel
        file: ../model/CpuModel/model.py
        cpu:
      - name: ImportedModel
        file: ../models/imported.pte
```

### `model:`

| `model:` | | Content |
|---|---|---|
| `clayer:` | Required | Path and filename of the `*.clayer.yml` file used to integrate the model into a CMSIS-style project. Its parent folder also receives the generated model files. |
| `zephyr:` | Optional | Enables generation of additional Zephyr integration files in the clayer path. A plain empty node and `zephyr: {}` are equivalent. |
| `input:` | Required | Generated sequence of model definitions. The converter also accepts `inputs` for compatibility with early multi-model drafts. |

Each model entry contains:

| Model entry | | Content |
|---|---|---|
| `name:` | Required | Unique C identifier used for the layer group, filenames, and embedded-array symbol. |
| `file:` | Required | Python model source or target-specific `.pte`, relative to the solution directory. |
| `cpu:` | Optional | Presence selects CPU-only conversion. The CPU type comes from `processor.type`. |
| `optimize:` | Optional | Per-model Vela optimization setting; invalid for CPU-only conversion. |
| `parameters:` | Optional | Mapping passed to the Python model functions. `input-shape` and `calibration-samples` are converted to Python-friendly names and types. |

The former single-model fields remain accepted for one compatibility release.

The matching `*.cbuild-pack.yml` file supplies the selected `PyTorch::ExecuTorch` pack
version.

## Output

The script uses the parent folder of the `*.clayer.yml` path specified by
`model.clayer` as its output folder. For example, with
`clayer: ai_layer/ai_layer.clayer.yml`, generated files are written to
`ai_layer/`.

For each model `<name>`, the script creates:

- `<name>.pte`;
- `<name>_pte.c` and `<name>_pte.h`, exporting unique embedded-array symbols;
  and
- `<name>_REPORT.md`, recording the input, conversion parameters, optimization,
  CPU fallback operators, CMSIS-NN requirement, and applicable Vela output.

The internal `.ai_layer_manifest.json` records generator-owned files so a
later full conversion can remove obsolete model artifacts without deleting
untracked or user-owned files.

The script also creates the `*.clayer.yml` file specified by `model.clayer`.
This canonical file contains one group per model and selects the shared
ExecuTorch runtime, required backends, and the union of CPU fallback-operator
components extracted from all `.pte` files. CPU-only layers omit `Backend
EthosU`; CMSIS-NN is selected when any fallback operator requires it.

When the `zephyr:` node is present, the script additionally creates
`CMakeLists.txt`, `components.conf`, and `operators.conf` for the Zephyr
application. These files are derived from `*.clayer.yml`; they do not inspect
the `.pte` files or calculate another operator union. CPU-only fragments omit
the Ethos-U backend.

The application root must append the two generated `.conf` files to
`EXTRA_CONF_FILE` before `find_package(Zephyr)`, then call
`add_subdirectory(ai_layer)` afterward. This lets the generated layer attach
every model C file directly and exactly once to Zephyr's `app` target.

All output files use LF line endings where applicable. The script stages the
complete generated layer in temporary storage and atomically replaces the
destination files only after export and output generation succeed. If Vela or
later generation fails, the previous AI layer remains intact. The generated
`*.cbuild-mlops.yml` input is never modified.

## Troubleshooting

### A global `PYTHONPATH` overrides the workspace environment

A global `PYTHONPATH` that contains another Python installation's
`site-packages` directory is placed before `.venv/Lib/site-packages` on
`sys.path`. The converter may then load incompatible global packages even
though it is running with the workspace's `.venv` interpreter. This can cause
import errors such as a missing symbol from `typing_extensions`.

Check which copy is being imported:

```console
.venv/Scripts/python -c "import typing_extensions; print(typing_extensions.__file__)"
```

On Windows PowerShell, remove `PYTHONPATH` only from the current shell before
running the converter:

```powershell
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
python model_conversion/create_ai_layer.py <solution>.cbuild-mlops.yml
```

On Linux or macOS, clear it only for the converter command:

```sh
PYTHONPATH= python model_conversion/create_ai_layer.py <solution>.cbuild-mlops.yml
```

The global setting can remain available in other shells for documentation or
repository-specific tooling.

## Tests

Run the unit tests from the repository root:

```console
python -m unittest discover -s model_conversion/test -p "test_*.py" -v
```

The tests mock model export and do not require Torch, Vela, an FVP, or target
hardware.
