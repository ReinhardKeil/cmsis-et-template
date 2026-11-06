# ExecuTorch model-conversion toolset specification

## Purpose and directory contract

`model_conversion/` is the complete host-side toolset for converting one or
more ML models into ExecuTorch programs for Arm Cortex-M, optionally with
Ethos-U acceleration. It generates one AI layer for integration into CMSIS- or
Zephyr-based projects. Each ML model keeps its own serialized and embedded
artifacts in the AI layer. The layer also contains the shared runtime and
backend components, the union of CPU fallback-operator requirements, and the
optional CMSIS-NN requirements needed to execute all models on one processor.

The directory above `model_conversion/` is the **workspace directory**. It
owns the shared `.venv`. The directory containing a selected
`*.cbuild-mlops.yml` is the **solution directory**. It owns the model inputs,
solution-specific configuration, pack metadata, and generated AI layer. A
workspace may contain multiple solution directories.

## Deliverables

### `README.md`

The README is the user entry point. It shall give the setup, conversion, and
test commands; explain the workspace/solution distinction; and link to this
specification for the complete directory contract.

### `create_ai_layer.py`

This is the model converter and AI-layer generator. Invoke it with exactly one
MLOps file and an optional model selection:

```console
python model_conversion/create_ai_layer.py <solution-dir>/<solution>.cbuild-mlops.yml
python model_conversion/create_ai_layer.py <solution-dir>/<solution>.cbuild-mlops.yml --model <name>
```

It shall locate `.venv` in the workspace directory and restart itself with
that environment's interpreter unless it is already running there. It shall
derive the solution directory from the MLOps file rather than from the current
working directory.

The YAML root shall contain `cbuild-mlops`. The converter uses these
layer-wide properties:

- `processor.type`: required Cortex-M type when `npu` is absent;
- `npu.type`: optional Ethos-U type; its absence selects CPU-only generation;
- `npu.macs`: optional MAC count, defaulting to 256;
- `vela.ini`: optional path relative to the solution directory;
- `vela.options`: generated command-line options;
- `model.clayer`: required output path relative to the solution directory;
- `model.zephyr`: optional empty node or mapping whose presence enables
  additional Zephyr integration output;
- `model.inputs`: generated sequence of model definitions described below.

The authored `mlops.model.input` sequence shall be emitted by CMSIS-Toolbox as
`cbuild-mlops.model.inputs`. Each entry uses this contract:

```yaml
cbuild-mlops:
  model:
    clayer: ai_layer/ai_layer.clayer.yml
    inputs:
      - name: TinyCNN
        file: ../TinyCNN-model/model.py
        optimize: size
        parameters:
          input-shape: 1x16x16x3
          calibration-samples: 2
      - name: CpuModel
        file: ../CpuModel/model.py
        cpu:
      - name: ImportedModel
        file: ../models/imported.pte
```

- `name` is required and unique. After validation and sanitization it supplies
  the layer group, artifact stem, and C identifier. Duplicate names, invalid
  identifiers, and name-to-identifier collisions are errors.
- `file` is required, is resolved relative to the solution directory, and must
  name a supported Python source or `.pte` input.
- `parameters` is an optional mapping passed to the Python model entry points.
  Unknown parameters shall fail with a model-qualified diagnostic instead of
  being ignored.
- Presence of the `cpu` key, including an empty `cpu:` value, selects CPU-only
  conversion for that model. The target comes from `processor.type`. Key
  presence, not YAML value truthiness, determines this behavior.
- Without `cpu`, a model uses the solution's NPU. If the solution has no NPU,
  every model is necessarily CPU-only.
- `optimize` is an optional per-model Vela optimization setting that supplements
  the layer-wide `vela.options`. It is invalid for a resolved CPU-only model.

Python inputs shall provide `get_model` and `get_calibration_inputs`. The
converter shall load them from the file named by that entry, pass the declared
parameters, quantize the exported graph, and calibrate it with all returned
samples. CPU-only conversion shall use explicit-layout Cortex-M quantization
and lowering and serialize operators implemented by CMSIS-NN. NPU conversion
shall delegate supported operations to Ethos-U and retain unsupported or
boundary operations as CPU fallback operators.

An input `.pte` shall be copied as an already-exported, target-specific model.
It remains subject to operator inspection and report generation. The converter
shall reject an imported `.pte` whose delegates or operators contradict its
declared CPU/NPU execution mode or the selected solution target.

The previous scalar/single-model `model` form shall remain accepted for one
release cycle as a compatibility input and shall normalize internally to one
model definition.

The accelerator configuration, system configuration, and memory mode are
layer-wide target properties. All NPU models in one AI layer shall use the same
effective values. Per-model system configurations or memory modes are not
supported.

Every Ethos-U/Vela invocation shall capture the complete textual report while
still showing it in the terminal. Its content shall be included in that model's
`<name>_REPORT.md`, not written as a layer-wide `VELA_SUMMARY.md`.

`--accelerator-config`, `--system-config`, and `--memory-mode` shall initialize
`EthosUCompileSpec`. Other Vela options shall be forwarded as extra flags.
`--config`, `--output-format`, and `--output-dir` are owned by the converter and
shall be rejected. Duplicate compile-spec options and unexpected positional
tokens shall also be rejected.

The matching `<solution>.cbuild-pack.yml` shall be read from the solution
directory. Its selected `PyTorch::ExecuTorch` version determines the pack used
to map serialized CPU operators to CMSIS components.

`model.zephyr` is a custom property under `mlops.model`; it shall be preserved
by CMSIS-Toolbox in the generated `*.cbuild-mlops.yml`. It shall not be placed
directly under `mlops`. A plain empty `zephyr:` node is equivalent to an empty
mapping. A non-empty value shall be a mapping so that future Zephyr-specific
parameters can be added without changing the node type.

The directory containing `model.clayer` shall be created as needed. For every
model it shall receive:

```text
<name>.pte
<name>_pte.c
<name>_pte.h
<name>_REPORT.md
```

`<name>_pte.c/.h` shall export unique `<name>_pte` and `<name>_pte_size`
symbols. The C array shall be aligned to 16 bytes.

`<name>_REPORT.md` shall be a concise, reproducible summary of how the model was
produced and optimized for the target CPU/MCU and, when applicable, NPU. It
shall identify the model input, distinguish generated and imported `.pte`
inputs, record conversion parameters and effective optimization settings, list
the model's CPU fallback operators, and state whether CMSIS-NN is required. An
Ethos-U report shall additionally contain the shared accelerator, system
configuration, and memory mode; the effective per-model Vela options; and the
complete Vela textual report in a fenced code block. A CPU-only report shall
have no Vela section. Reports shall use UTF-8 and LF line endings.

The directory shall also receive the `*.clayer.yml` file named by
`model.clayer`. It is the canonical build description and shall contain one
group per model with that model's C and header files, one shared ExecuTorch
runtime component set, every backend needed by at least one model, and the
union of CPU fallback-operator components across all model `.pte` files.
Repeated components shall be de-duplicated and sorted. Select Backend EthosU
once if any model is delegated. Add `ARM::CMSIS-NN` without a version,
`CMSIS:NN Lib`, and required defines when any selected CPU fallback operator
requires CMSIS-NN.

When `model.zephyr` is present, the directory shall additionally receive
`CMakeLists.txt`, `components.conf`, and `operators.conf`. These files are a
projection of the completed `*.clayer.yml`: they shall not inspect `.pte` files
or independently calculate an operator union. The CMake file shall add every
model C file directly and exactly once to Zephyr's `app` target for every
compiler. The configuration fragments shall enable the same runtime, backends,
and fallback-operator components as the CMSIS layer.

The consuming Zephyr application's root CMake file shall append the generated
configuration fragments to `EXTRA_CONF_FILE` before `find_package(Zephyr)` and
shall add the generated layer directory afterward. The generated layer shall
not defer source ownership or expose the model through interface sources.
CPU-only configuration fragments shall not enable the Ethos-U backend.

### Fallback-operator extraction and union

The current `.pte` artifact of every declared model is the authoritative input
for fallback-operator selection. `<name>_REPORT.md` is generated output and
shall never be used as operator-selection input.

For each `.pte`, the converter shall call ExecuTorch's `gen_oplist` in staging
to produce a temporary, per-model `selected_operators.yaml`. The per-model list
shall supply the fallback operators recorded in that model's report. The
converter shall pass all per-model lists to `gen_all_oplist`, which produces a
temporary combined `selected_operators.yaml`. It shall map the combined YAML
operator keys to components in the resolved pack PDSC and write the resulting
union to the canonical `*.clayer.yml`.

The converter shall not search raw `.pte` bytes or use kernel-definition
`merge_yaml` as a multi-model union. An operator with no matching pack
component and a selected Zephyr operator without a matching Kconfig option are
errors. Temporary operator YAML files shall not be installed in the AI layer.

The installed ExecuTorch Python package version shall match the resolved
`PyTorch::ExecuTorch` pack version. A mismatch shall fail before model export or
operator inspection.

Without `--model`, the converter shall convert every declared model and rebuild
all layer-wide metadata. With `--model <name>`, it shall convert only the named
model. It shall use the newly staged `.pte` for that model and the existing
`.pte` artifacts for all unchanged models, run `gen_oplist` on every one, and
rebuild the selected model's report, the operator union, `*.clayer.yml`, and
the optional Zephyr projection. Reports and other per-model artifacts belonging
to unchanged models shall be retained. An unknown model name is an error.
Partial generation shall fail without changing the layer if another declared
`.pte` is missing, cannot be inspected, or is stale for a different target or
export configuration.

All generated artifacts shall first be prepared in temporary storage. Only
after model export/import, Vela compilation, operator union, component
discovery, report generation, and output generation succeed shall the script
atomically replace destination files. A failure shall preserve the previous
complete AI layer and shall not leave partial destination files. Obsolete
generated artifacts may be removed only when recorded in the previous
`.ai_layer_manifest.json` generator manifest; untracked or user-owned files
shall not be deleted. The generated MLOps file shall never be modified.

### `setup_venv.py`

This is the platform-independent environment setup implementation. It shall
create or repair `<workspace>/.venv`, never a solution-local environment. It
shall support Python 3.10 through 3.13 and reject unsupported versions before
dependency installation.

`--recreate` shall rebuild the environment. `--uv` shall use uv for environment
creation and package installation. `--python VERSION` shall select the uv
Python version and shall require `--uv`.

The setup shall install `requirements.txt` first and then install
`requirements-arm-tosa.txt` with `--no-deps`. It shall finish with an import
test of the ExecuTorch Arm quantizer, `executorch.codegen.tools.gen_oplist`, and
`executorch.codegen.tools.gen_all_oplist`.

### `setup_venv.sh`

This is the Linux and macOS entry point. It shall locate `setup_venv.py`
relative to itself, forward all arguments, use `${PYTHON:-python3}` for the
standard setup, and use an isolated uv launcher when `--uv` is requested.

### `setup_venv.bat`

This is the Windows entry point. It shall locate `setup_venv.py` relative to
itself, forward all arguments, use `%PYTHON%` when provided and `python`
otherwise, and use an isolated uv launcher when `--uv` is requested.

### `requirements.txt`

This is the primary dependency manifest for model conversion. Versions shall
remain compatible with the ExecuTorch Arm backend and the supported Python
range. It is the source of truth for the coordinated ExecuTorch, PyTorch,
TorchAO, and Vela versions. The ExecuTorch version shall match the version of
the `PyTorch::ExecuTorch` pack that consumes the generated `.pte` files.
`gen_oplist` and `gen_all_oplist` come from ExecuTorch, and TorchGen comes from
PyTorch; they require no separate dependency entries.

### `requirements-arm-tosa.txt`

This is the separately installed TOSA serializer manifest. It shall contain
the serializer and required FlatBuffers pin used by the ExecuTorch Arm
backend. It is deliberately installed with `--no-deps` so package metadata
cannot replace the dependency versions selected by the primary environment.

### `model_conversion-specification.md`

This document is the normative directory and behavior specification. Changes
to any deliverable in `model_conversion/` shall be reflected here.

### `test/test_create_ai_layer.py`

This is the unit-test suite for the converter. It shall cover workspace `.venv`
selection, solution-relative path handling, multi-model MLOps parsing, model
selection, per-model CPU/NPU resolution, imported `.pte` validation, Vela
options, package/pack version matching, per-model and combined operator-list
generation, operator component selection, reports, atomic updates, and CMSIS
and Zephyr artifacts. Unit tests shall mock model export, `gen_oplist`, and
`gen_all_oplist.main` so they do not require Torch, Vela, an FVP, or target
hardware. One integration test shall exercise operator-list generation with
the pinned ExecuTorch package.

Run it from the workspace directory:

```console
python -m unittest discover -s model_conversion/test -p "test_*.py" -v
```

## Acceptance criteria

- Invocation from any current directory locates the workspace `.venv` and the
  selected solution's files correctly.
- Multiple solutions in workspace subdirectories share `.venv` while keeping
  their models, configuration, metadata, and output independent.
- Without `--model`, every declared model is converted. With `--model <name>`,
  only that model is converted while the layer-wide operator union and metadata
  continue to include every declared model.
- Two models with overlapping CPU fallback operators produce one copy of each
  runtime and operator component in CMSIS and Zephyr metadata.
- A CPU-only model and an Ethos-U model can share one layer; it selects their
  combined CPU/CMSIS-NN requirements, the Ethos-U backend once, and both model
  arrays without symbol collisions.
- All NPU models use the same effective accelerator configuration, system
  configuration, and memory mode.
- Python and imported `.pte` inputs produce model-qualified reports that record
  provenance, parameters, optimization, fallback operators, and applicable
  Vela output.
- Unknown model names, duplicate names, identifier collisions, missing inputs,
  unsupported file types, and target-incompatible imported `.pte` inputs fail
  before replacing the complete layer.
- Both `--name value` and `--name=value` Vela option forms work.
- Invalid and duplicate options terminate with a clear error.
- The resolved pack version follows the exact selector when available and
  mismatches with the installed ExecuTorch package are rejected.
- Model parameters are converted to the types expected by each Python model.
- Generated C, header, report, layer, and Zephyr text are deterministic for
  identical inputs and tool versions.
- Both Ethos-U and CPU-only Cortex-M generation select their target-specific
  runtime and operator components.
- A present `model.zephyr` node generates complete Zephyr metadata; without it,
  no Zephyr metadata is generated. The Zephyr metadata is derived from
  `*.clayer.yml` and does not recalculate the operator union.
- Failed full or partial conversion leaves the previous complete layer intact
  and never removes untracked or user-owned files.
- The complete unit-test suite passes without target hardware.

