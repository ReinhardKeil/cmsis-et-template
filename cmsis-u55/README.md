# CMSIS Cortex-M with Ethos-U55 quick start

This workflow builds the template application for the Corstone-300 FVP with an
Ethos-U55-128 NPU. Its CMSIS solution is
`cmsis-u55/cmsis-u55.csolution.yml`.

Complete the workspace-level [prerequisites](../README.md#prerequisites) first.
Commands below run from the workspace root.

## Keil Studio for VS Code

1. Create a repository from the GitHub template, clone it, and open its root
   directory in VS Code.
2. In the CMSIS view, select **... > Select Active Solution from Workspace**,
   then select `cmsis-u55/cmsis-u55.csolution.yml`.
3. Select **Terminal > Run Task > Create Python environment (uv)**. The pip
   variant can be used when a compatible Python `>=3.10,<3.14` is installed.
4. Use the CMSIS action buttons to build the application, then select **Run** or
   **Debug**. Keil Studio starts the Corstone-300 Ethos-U55 FVP automatically.

The generated AI layer is committed, so the initial build does not require a
model export. After changing the model or its target settings, run `cbuild
setup` and select **Terminal > Run Task > Create AI layer** before rebuilding.

A successful run ends with:

```text
Test_result: PASS
```

## Command-line build

The shell must have the tools from `vcpkg-configuration.json` on `PATH`.

### 1. Create the shared Python environment

On Linux or macOS:

```bash
./model_conversion/setup_venv.sh
```

On Windows:

```powershell
.\model_conversion\setup_venv.bat
```

The script creates `.venv/` at workspace level. With
[uv](https://docs.astral.sh/uv/), append `--uv --python 3.12` to either command.

### 2. Generate the MLOps information

Enable the `SSE-300-U55` target-set with `--active`:

```bash
cbuild setup cmsis-u55/cmsis-u55.csolution.yml --active SSE-300-U55 --packs
```

This writes `cmsis-u55/cmsis-u55.cbuild-mlops.yml` with the processor,
Ethos-U55, Vela, simulator, and AI-layer settings.

## Configuring the Ethos-U target

Follow the
[CMSIS-Ethos-U integration guide](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html)
when adapting this template. The default U55-128 configuration is represented
by the solution's `mlops:` settings, `board/Board-U55.clayer.yml` with
`ETHOSU_MACS: 128`, and `board/fvp_config_u55.txt` with
`ethosu.num_macs=128`. The included U65-256 board and FVP configurations use
256 MACs in both corresponding files.

The guide's
[256-MAC FVP configuration example](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html#configure-256-macs-for-fvp-simulation)
shows the settings that must remain aligned. If the NPU configuration changes,
update the solution, Board layer, and FVP together, then repeat model generation
so Vela compiles the command stream for the matching accelerator.

### 3. Create the AI layer

```bash
python3 model_conversion/create_ai_layer.py cmsis-u55/cmsis-u55.cbuild-mlops.yml
```

On Windows, use `python`. The conversion tool uses the shared `.venv/` and
`model/`, then writes target-specific output to `cmsis-u55/ai_layer/`.

### 4. Build the application

Keep the same target-set active for the build:

```bash
cbuild cmsis-u55/cmsis-u55.csolution.yml --active SSE-300-U55
```

The resulting image is under
`cmsis-u55/out/cmsis-u55/SSE-300-U55/Debug/`.

### 5. Run on the FVP

```bash
FVP_Corstone_SSE-300_Ethos-U55 \
    -f cmsis-u55/board/fvp_config_u55.txt --simlimit 60 \
    -a cmsis-u55/out/cmsis-u55/SSE-300-U55/Debug/cmsis-u55.axf
```

For more background, see the workspace documentation for the
[MLOps flow](../documentation/mlops-flow.md) and
[model-conversion tools](../model_conversion/README.md).
