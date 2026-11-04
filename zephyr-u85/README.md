# Zephyr Cortex-M with Ethos-U85

This template application runs an ExecuTorch model with Ethos-U85 acceleration
under Zephyr 4.5. It contains target configurations for an Alif Ensemble E8
AI/ML AppKit and the Corstone-320 FVP.

> [!NOTE]
> The Alif E8 hardware target is included temporarily for development and
> validation. It will be removed from the final template.

> [!IMPORTANT]
> This Zephyr template does not use the `ARM::CMSIS-Ethos-U` pack. The Ethos-U
> driver is part of upstream Zephyr 4.5 and uses Zephyr's `hal_ethos_u` module.
> `PyTorch::ExecuTorch` remains an external module and supplies the ExecuTorch
> runtime and Ethos-U backend.

The current solution uses the `PyTorch::ExecuTorch` 1.6.0 Zephyr preview
module. Its path is supplied to Zephyr through `EXTRA_ZEPHYR_MODULES`.
The Zephyr targets use the Arm GNU Toolchain (`compiler: GCC`) by default.
Their CMake and Kconfig configuration also carries the AC6 6.24 compatibility
needed by the ExecuTorch preview pack when the solution compiler is changed to
AC6.

## Quick Start

### 1. Prepare Zephyr 4.5

Create or select a Zephyr 4.5 workspace and fetch the modules needed by the
targets:

```sh
west init zephyr-workspace --mr v4.5.0
cd zephyr-workspace
west config manifest.project-filter -- +zephyr,+cmsis,+cmsis_6,+hal_ethos_u
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
- `AppKit-E8` for an Alif Ensemble E8 AI/ML AppKit using its HP Cortex-M55 and
  256-MAC Ethos-U85.

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
Arm FVP debug configuration. For `AppKit-E8`, connect the board's J-Link probe
and start the generated debug configuration.

The application initializes the Zephyr Ethos-U device, loads the embedded
ExecuTorch program from `ai_layer/model_pte.c`, runs inference, and reports the
result on the console.

### Configure with menuconfig

Build the desired target once with the CMSIS Solution extension. From the
repository root, open a terminal using the same tools environment and run:

```powershell
cmake --build zephyr-u85/out/Zephyr_app/Corstone-320-FVP/Debug --target menuconfig
```

Use `--target guiconfig` for the graphical interface. For the physical board,
replace `Corstone-320-FVP` with `AppKit-E8` after building that target once.
Each build directory has its own configuration.

In menuconfig, press `/` to search for an option such as `CMSIS_EXECUTORCH`,
press `?` for its dependencies, and press `Q`, then `Y`, to save. Changes are
written to the build directory's `zephyr/.config`. Copy settings that must
survive a clean build into `prj.conf`; files under `ai_layer/*.conf` are
generated and may be overwritten.

### Alif E8 validation output

The application produces the following output when it runs successfully on the
Alif Ensemble E8 AI/ML AppKit:

```text
Hello World! ensemble_e8_ak/ae822fa0e5597bs0/rtss_hp
Ethos-U85 (NPU-HG) driver communication check
Device: npu@49042000, base: 0x49042000, IRQ: 366
Driver version: 2.0.0
Hardware architecture: 2.0.0
Hardware revision: 0.0 (status 1)
Product revision: 7
Command stream version: 1
Custom DMA: 0
MACs per cycle: 256
Ethos-U85 driver communication OK (ID/configuration read; no inference).
ExecuTorch Ethos-U85-256 application: 8656 byte model
Input shape: [1, 16, 16, 3]
I: Optimizer release nbr: 0 patch: 1
I: Optimizer config. product=2, cmd_stream_version=1, macs_per_cc=8, num_axi_ext=1, num_axi_sram=2, custom_dma=0
I: Optimizer config. arch version=2.0.0
I: Ethos-U config. product=2, cmd_stream_version=1, macs_per_cc=8, num_axi_ext=1, num_axi_sram=2, custom_dma=0
I: Ethos-U. arch version=2.0.0
I: handle_command_stream: cmd_stream=0x80231f90, cms_length 216 words (864 bytes)
I: Soft reset ethos-u85
Output: 10 element(s): -0.0142 0.0206 0.0601 -0.0886 0.0791 0.0522 -0.0301 -0.0775 -0.2800 -0.0095
Test_result: PASS
```

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
