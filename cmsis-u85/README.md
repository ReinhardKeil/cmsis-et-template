# CMSIS Cortex-M with Ethos-U85 quick start

This CMSIS template deploys ML models on Cortex-M targets with Ethos-U85
acceleration using ExecuTorch.

The supplied configuration runs on the Corstone-320 FVP with an Ethos-U85-256
NPU. Its solution is `cmsis-u85/cmsis-u85.csolution.yml`.

Commands below run from the workspace root.

The generated AI layer is committed, so the initial build does not require a
model export. After changing the model or its target settings, run `cbuild
setup` and select **Terminal > Run Task > Create AI layer** before rebuilding.

A successful run prints the Ethos-U configuration, output logits, and a pass
result:

```text
Ethos-U version info:
    Arch:       v2.0.0
    MACs/cc:    256
    Cmd stream: v1
ExecuTorch Ethos-U85-256 application: 8720 byte program
Output: 10 element(s): -0.0142 0.0206 0.0601 -0.0886 0.0791 0.0522 -0.0301 -0.0775 -0.2800 -0.0095
Test_result: PASS
```

On macOS, where Arm ships no native FVP, configure the solution to use
`${workspaceFolder}/.vscode/fvp.sh` and use `.vscode/launch.json.mac` as the
launch configuration. Docker Desktop must be running.

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

Enable the `SSE-320-U85` target-set with `--active`:

```bash
cbuild setup cmsis-u85/cmsis-u85.csolution.yml --active SSE-320-U85 --packs
```

This writes `cmsis-u85/cmsis-u85.cbuild-mlops.yml` with the processor,
Ethos-U85, Vela, simulator, and AI-layer settings.

## Configuring the Ethos-U target

Follow the
[CMSIS-Ethos-U integration guide](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html)
when adapting this template. Its U85-256 configuration must remain consistent
between the solution's `mlops:` and Vela settings,
`board/Corstone-320/Board-U85.clayer.yml` (`ETHOSU_MACS: 256`), and
`board/Corstone-320/fvp_config.txt` (`mps4_board.subsystem.ethosu.num_macs=256`). Memory
placement and attributes must also match the selected hardware.

See the guide's
[256-MAC FVP configuration example](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html#configure-256-macs-for-fvp-simulation)
for the coordinated Board-layer, MLOps, Vela, and FVP changes. After changing
the configuration, repeat model generation so Vela compiles the command stream
for the matching accelerator, then rebuild the application.

### 3. Create the AI layer

```bash
python3 model_conversion/create_ai_layer.py cmsis-u85/cmsis-u85.cbuild-mlops.yml
```

On Windows, use `python`. The conversion tool uses the shared `.venv/` and
`model/`, then writes target-specific output to `cmsis-u85/ai_layer/`.

### 4. Build the application

Keep the same target-set active for the build:

```bash
cbuild cmsis-u85/cmsis-u85.csolution.yml --active SSE-320-U85
```

The resulting image is
`cmsis-u85/out/cmsis-u85/SSE-320-U85/Debug/cmsis-u85.axf`.

### 5. Run on the FVP

```bash
FVP_Corstone_SSE-320 \
    -f cmsis-u85/board/Corstone-320/fvp_config.txt --simlimit 60 \
    -a cmsis-u85/out/cmsis-u85/SSE-320-U85/Debug/cmsis-u85.axf
```

On macOS, call `.vscode/fvp.sh` with the same arguments to run the FVP in
Docker.

For more background, see the workspace documentation for the
[MLOps flow](../documentation/mlops-flow.md) and
[model-conversion tools](../model_conversion/README.md).
