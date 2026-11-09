# ML model to ExecuTorch program conversion

`create_ai_layer.py` reads the target, model inputs, and output locations from
a CMSIS-Toolbox `*.cbuild-mlops.yml` file and creates the AI layer for the
selected CMSIS or Zephyr solution. It converts Python model sources or imports
and lowers quantized `.pt2` archives, or validates existing target-specific
`.pte` programs. One AI layer targets exactly one CPU, optionally with Ethos-U
acceleration.

The top-level [README](../README.md#how-ml-model-generation-works) explains the
workflow and generated AI-layer structure.

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
program files. `gen_oplist` and `gen_all_oplist` are included with ExecuTorch;
TorchGen is included with PyTorch.

Use `--recreate` to rebuild an existing environment. The optional `--uv` and
`--python VERSION` arguments select uv and a supported Python version:

```console
./model_conversion/setup_venv.sh --uv --python 3.12 --recreate
```

## Convert models

Pass the MLOps file belonging to the selected solution. For a solution in the
workspace root:

```console
cbuild setup cmsis-u85/cmsis-u85.csolution.yml --active SSE-320-U85 --packs      # create *.cbuild-mlops.yml
python model_conversion/create_ai_layer.py cmsis-u85/cmsis-u85.cbuild-mlops.yml  # create ExecuTorch programs
cbuild cmsis-u85/cmsis-u85.csolution.yml --active SSE-320-U85                    # build application
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

Without additional options, every declared input is processed. To update one
input while retaining the others, use:

```console
python model_conversion/create_ai_layer.py solutions/demo/demo.cbuild-mlops.yml --input TinyCNN
```

The selected input is processed again. The generator then inspects every
declared `.pte` program file and rebuilds the layer-wide fallback-operator
union and metadata. A per-input report is output only; it is never
operator-selection input.

## Input

For an Ethos-U target, the `*.cbuild-mlops.yml` file provides the `npu:` and
`vela:` configuration described in the
[CMSIS-Toolbox MLOps documentation](https://open-cmsis-pack.github.io/cmsis-toolbox/YML-Input-Format/#mlops-management).
For a CPU-only target, omit `npu:`. On a mixed target, an empty `cpu:` key in an
input entry selects CPU lowering for that input. Accelerator, system
configuration, and memory mode are shared by every Ethos-U program in the
layer.

Declare inputs under `mlops.model.input` in the `*.csolution.yml` file.
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
        file: ../models/cpu-model.pt2
        cpu:
      - name: ImportedModel
        file: ../models/imported.pte
```

### `model:`

| `model:` | | Content |
|---|---|---|
| `clayer:` | Required | Path and filename of the `*.clayer.yml` file used to integrate the model into a CMSIS-style project. Its parent folder also receives the generated program artifacts. |
| `zephyr:` | Optional | Enables generation of additional Zephyr integration files in the clayer path. A plain empty node and `zephyr: {}` are equivalent. |
| `input:` | Required | Generated sequence of input-artifact definitions. |

The `input:` node contains a list of named input artifacts.

| `input:` | | Content |
|---|---|---|
| `- name:` | Required | Unique C identifier used for the layer group, filenames, and embedded-array symbol. |
| &nbsp;&nbsp;&nbsp;`file:` | Required | `.py`, `.pt2`, or `.pte` file relative to the solution directory. The file extension selects the workflow. |
| &nbsp;&nbsp;&nbsp;`cpu:` | Optional | Presence selects CPU lowering for `.py` and `.pt2`, or asserts CPU-only execution for `.pte`. The CPU type comes from `processor.type`. |
| &nbsp;&nbsp;&nbsp;`optimize:` | Optional | Per-input Vela optimization setting for NPU-resolved `.py` and `.pt2`; invalid for CPU-resolved inputs and `.pte`. |
| &nbsp;&nbsp;&nbsp;`parameters:` | Optional | `.py` only. Mapping passed to accepting Python workflow functions. `input-shape` and `calibration-samples` are converted to Python-friendly names and types. |

The file extension selects the processing workflow:

| Input | Role | `create_ai_layer` action |
|---|---|---|
| `.pt2` | Recommended: lowering input | Validate, backend-lower, serialize as `.pte`, and integrate. |
| `.pte` | Optional: prebuilt deployment artifact | Inspect, validate, and integrate unchanged. |
| `.py` | Convenience: model-production input | Export, quantize, backend-lower, serialize, and integrate. |

An input file is a container boundary, not necessarily one model. Each
input entry owns one output `.pte` container. A `.pt2` may contain several
named, exported PyTorch programs; they become named methods in that output.
Separate input entries are not implicitly combined.

### `.pt2` input

This is the recommended boundary for model-specific preparation. The archive
contains one or more named, already-quantized PyTorch programs together with
their input signatures, but no target-specific lowering. `create_ai_layer`
validates every named program, lowers all of them for the selected CPU or
Ethos-U configuration, and serializes them into one `.pte`. It never quantizes
or requantizes `.pt2`.

### `.pte` input

This optional workflow integrates a prebuilt deployment artifact. The program
is deserialized and its methods, delegates, and Ethos-U compile specification
are checked against the selected target. It is integrated unchanged and is
never passed through Vela or another lowering pass. Invalid, portable-only NPU,
unknown-backend, or target-mismatched programs are rejected.

### `.py` input

This convenience workflow performs export, built-in quantization, lowering,
and serialization. The module must define:

- `get_model(...) -> torch.nn.Module`;
- `get_example_inputs(...) -> (args, kwargs)`; and
- `get_calibration_inputs(...) -> [(args, kwargs), ...]`.

Example inputs define the export signature and are deliberately separate from
representative PTQ calibration cases. Both positional and keyword inputs are
supported. Use `.pt2` for custom quantization, QAT, specialized calibration, or
producer-owned accuracy validation.

## Output

The script writes generated files to the directory containing the
`*.clayer.yml` file specified by `model.clayer`. For example, with
`clayer: ai_layer/ai_layer.clayer.yml`, generated files are written to
`ai_layer/`.

```text
ai_layer/
|-- ai_layer.clayer.yml              # canonical build description
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

For each input `<name>`, the script creates:

- `<name>.pte`, the serialized ExecuTorch program;
- `<name>_pte.c` and `<name>_pte.h`, exporting unique embedded-array symbols;
  and
- `<name>_REPORT.md`, recording the input, conversion parameters, optimization,
  CPU fallback operators, CMSIS-NN requirement, and applicable Vela output.

The generator replaces the files produced by the current run. It does not
delete other files from the output directory; obsolete artifacts from removed
inputs can be deleted by the user explicitly.

The script also creates the `*.clayer.yml` file specified by `model.clayer`.
This canonical file contains one group per input and selects the shared
ExecuTorch runtime, required backends, and the union of CPU fallback-operator
components extracted from all `.pte` program files. CPU-only layers omit `Backend
EthosU`; CMSIS-NN is selected when any fallback operator requires it.

When the `zephyr:` node is present, the script additionally creates
`CMakeLists.txt`, `components.conf`, and `operators.conf` for the Zephyr
application. These files and `*.clayer.yml` are rendered from the same
validated layer data.

Add the generated configuration fragments before Zephyr processes the
configuration, then add the AI-layer directory after creating the application:

```cmake
# Select the AI layer's Kconfig fragments before Zephyr processes configuration.
list(APPEND EXTRA_CONF_FILE
  "${CMAKE_CURRENT_SOURCE_DIR}/ai_layer/components.conf"
  "${CMAKE_CURRENT_SOURCE_DIR}/ai_layer/operators.conf"
)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(zephyr_m)

add_subdirectory(ai_layer)
```

All output files use LF line endings where applicable. The generated
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

Most tests mock model export and do not require Torch or Vela. An
environment-backed PTE validation test is skipped by a host interpreter without
ExecuTorch; run the suite with `.venv/Scripts/python.exe` on Windows or
`.venv/bin/python` on Linux/macOS to generate real portable and Ethos-U PTE
files and exercise malformed and target-mismatch rejection. No test requires an
FVP or target hardware.
