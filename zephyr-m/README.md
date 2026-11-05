# Zephyr Cortex-M55 CPU-only

This template runs the TinyCNN ExecuTorch model entirely on the HP Cortex-M55
of the Alif Ensemble E8 AI/ML AppKit. It uses the ExecuTorch Cortex-M backend
and the CMSIS-NN module supplied by Zephyr, without an Ethos-U NPU or Vela.

## Prepare Zephyr 4.5

Create or select a Zephyr 4.5 workspace and fetch the modules required by the
CPU-only application:

```sh
west init zephyr-workspace --mr v4.5.0
cd zephyr-workspace
west config manifest.project-filter -- +zephyr,+cmsis,+cmsis_6,+cmsis-dsp,+cmsis-nn
west update
west zephyr-export
```

Open this repository in VS Code from a shell where the Zephyr Python
environment is active. Install the prerequisites in the
[top-level README](../README.md#prerequisites), then open
`zephyr-m/zephyr-m.csolution.yml` with the CMSIS Solution extension.

## Generate and build

From the workspace root:

```sh
cbuild setup zephyr-m/zephyr-m.csolution.yml --active AppKit-E8
python model_conversion/create_ai_layer.py zephyr-m/zephyr-m.cbuild-mlops.yml
cbuild zephyr-m/zephyr-m.csolution.yml --active AppKit-E8
```

The solution sets `model.cpu: Cortex-M55` because the Alif device pack describes
the E8's available NPU even when an application does not use it. This value is
the processor name defined by Open-CMSIS-Pack. The generator applies the
Cortex-M quantizer and lowering passes for that processor and generates Kconfig
fragments for only the required ExecuTorch operators. `CONFIG_CMSIS_NN=y`
enables Zephyr's CMSIS-NN module; the selected Cortex-M operators then select
the required CMSIS-NN source groups.

Connect the AppKit's supported debug probe and use **CMSIS: Load & Run** or
**CMSIS: Load & Debug**. A successful inference ends with:

```text
Test_result: PASS
```

The AppKit image starts at MRAM offset `0x200000`, after the platform firmware.
No Ethos-U devicetree node, driver, `hal_ethos_u` module, or Vela configuration
is used.

With GCC 15, the application keeps CMSIS-NN's `int32_t*` versus MVE intrinsic
`long*` compatibility diagnostic as a warning. Both types are 32-bit on this
target; other compiler diagnostics remain unchanged.
