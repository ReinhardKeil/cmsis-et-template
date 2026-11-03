# Zephyr Cortex-M with Ethos-U85

This template application runs an ExecuTorch model with Ethos-U85 acceleration
under Zephyr 4.5. It contains target configurations for an Alif Ensemble E8
Development Kit and the Corstone-320 FVP.

> [!IMPORTANT]
> This Zephyr template does not use the `ARM::CMSIS-Ethos-U` pack. The Ethos-U
> driver is part of upstream Zephyr 4.5 and uses Zephyr's `hal_ethos_u` module.
> `PyTorch::ExecuTorch` remains an external module and supplies the ExecuTorch
> runtime and Ethos-U backend.

The current solution uses the `PyTorch::ExecuTorch` 1.6.0 Zephyr preview
module. Its path is supplied to Zephyr through `EXTRA_ZEPHYR_MODULES`.
The Zephyr targets use the Arm GNU Toolchain (`compiler: GCC`).

## Quick Start

### 1. Prepare Zephyr 4.5

Create or select a Zephyr 4.5 workspace and fetch the modules needed by the
targets:

```sh
west init zephyr-workspace --mr v4.5.0
cd zephyr-workspace
west config manifest.project-filter -- +zephyr,+cmsis,+cmsis_6,+hal_ethos_u,+hal_alif
west update
west zephyr-export
```

Open this template repository in VS Code from a shell where the Zephyr Python
environment is active. Install the prerequisites listed in the
[top-level README](../README.md#prerequisites), then open
`zephyr-u85/zephyr-u85.csolution.yml` with the CMSIS Solution extension.

### 2. Select a target-set

Choose one of these CMSIS-Toolbox target-sets:

- `Corstone-320-FVP` for simulation.
- `DevKit-E8` for an Alif Ensemble E8 Development Kit.

In command-line operations, enable the selection with `--active`, for example
`cbuild zephyr-u85/zephyr-u85.csolution.yml --active Corstone-320-FVP`.

The solution passes the selected Zephyr board and overlay to `west`. Its
`mlops:` node describes the Ethos-U85-256 and Vela memory configuration used to
generate the model. `fvp_config.txt` configures the Corstone-320 FVP with the
matching 256-MAC NPU, enables the Ethos-U fast functional mode, and routes
Zephyr's UART0 console to standard output. Fast mode is intended for functional
validation; its reported NPU timing is not cycle-accurate.

### 3. Build and run

Run **CMSIS: Build** in VS Code. For `Corstone-320-FVP`, start the generated
Arm FVP debug configuration. For `DevKit-E8`, connect the board and start the
generated pyOCD debug configuration.

The application initializes the Zephyr Ethos-U device, loads the embedded
ExecuTorch program from `ai_layer/model_pte.c`, runs inference, and reports the
result on the console.

## Configuring the Ethos-U target

The configuration-consistency guidance in the
[CMSIS-Ethos-U integration guide](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html)
also applies to this Zephyr template, although the `ARM::CMSIS-Ethos-U` pack
does not. The default U85-256 target is described by the solution's `mlops:`
settings, `CONFIG_ETHOS_U85_256=y` in `prj.conf`, the target devicetree overlay,
and `mps4_board.subsystem.ethosu.num_macs=256` in `fvp_config.txt`.

The guide provides a
[256-MAC FVP configuration example](https://arm-software.github.io/CMSIS-Ethos-U/latest/integration/index.html#configure-256-macs-for-fvp-simulation).
For Zephyr, apply the equivalent driver configuration through Kconfig and
devicetree. If the NPU configuration changes, keep those settings and the FVP
aligned, regenerate the AI layer so Vela targets the matching accelerator, and
rebuild the application.

## Adapting the template

- Change board, device, overlay, and debugger settings in
  `zephyr-u85.csolution.yml` for the target hardware.
- Change `prj.conf` for application-wide Zephyr settings and use a target
  overlay for the Ethos-U device description when the board does not provide
  one.
- Replace the model through the shared `model/` and `model_conversion/`
  workflow. The generated `ai_layer/components.conf` and `operators.conf`
  select only the required ExecuTorch features.

Do not add `ARM::CMSIS-Ethos-U` to this solution. That pack is used only by the
CMSIS workflow templates in this repository.
