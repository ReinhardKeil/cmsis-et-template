# Zephyr Cortex-M with Ethos-U55

This template application runs an ExecuTorch model on the 256-MAC Ethos-U55
integrated in the Alif Ensemble E8 AI/ML AppKit. It also provides a matching
Corstone-300 FVP target for simulation.

> [!IMPORTANT]
> This Zephyr template does not use the `ARM::CMSIS-Ethos-U` pack. The Ethos-U
> driver is part of upstream Zephyr 4.5 and uses Zephyr's `hal_ethos_u` module.
> `PyTorch::ExecuTorch` supplies the ExecuTorch runtime and Ethos-U backend.

The solution uses the Arm GNU Toolchain (`compiler: GCC`).

## Quick Start

### 1. Prepare Zephyr 4.5

Create or select a Zephyr 4.5 workspace and fetch the required modules:

```sh
west init zephyr-workspace --mr v4.5.0
cd zephyr-workspace
west config manifest.project-filter -- +zephyr,+cmsis,+cmsis_6,+hal_ethos_u
west update
west zephyr-export
```

Open this repository in VS Code from a shell where the Zephyr Python
environment is active. Install the prerequisites in the
[top-level README](../README.md#prerequisites), then open
`zephyr-u55/zephyr-u55.csolution.yml` with the CMSIS Solution extension.

### 2. Select a target-set

- `Corstone-300-FVP` for simulation.
- `AppKit-E8` for an Alif Ensemble E8 AI/ML AppKit using its HP Cortex-M55 and
  256-MAC Ethos-U55.

Enable a command-line selection with `--active`, for example:

```sh
cbuild zephyr-u55/zephyr-u55.csolution.yml --active Corstone-300-FVP
```

The solution describes the E8's Ethos-U55-256 and HP-core Vela memory
configuration. `fvp_config.txt` selects the same 256-MAC NPU and enables fast
functional simulation; reported NPU timing is therefore not cycle-accurate.

### 3. Generate, build, and run

After selecting a target-set, run solution setup and the workspace **Create AI
layer** task so the model is compiled for Ethos-U55-256. Then run **CMSIS:
Build**.

Start the generated Arm FVP configuration for `Corstone-300-FVP`. For
`AppKit-E8`, connect the board's supported debug probe and start the generated
pyOCD configuration. The application prints the detected NPU configuration,
performs inference, and ends with `Test_result: PASS` when validation succeeds.

## Configuring the Ethos-U target

Follow the
[CMSIS-Ethos-U integration guide](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html)
when adapting the hardware configuration. Its consistency principles apply to
Zephyr even though this template does not use the `ARM::CMSIS-Ethos-U` pack.

The U55-256 target is represented by the solution's `mlops:` settings,
`CONFIG_ETHOS_U55_256=y` in `prj.conf`, the enabled `ethosu0` devicetree node,
and `ethosu.num_macs=256` in `fvp_config.txt`. The guide's
[256-MAC FVP example](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html#configure-256-macs-for-fvp-simulation)
explains why these settings and the Vela accelerator selection must agree.

After any NPU change, regenerate the AI layer and rebuild the application.
