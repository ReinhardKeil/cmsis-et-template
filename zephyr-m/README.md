# Zephyr Cortex-M CPU-only

This Zephyr template deploys ML models on Cortex-M targets using ExecuTorch
and Zephyr's CMSIS-NN module for CPU-only inference. It does not require an
Ethos-U NPU or Vela compilation.

The supplied configurations run TinyCNN on a Cortex-M55 using the Alif
Ensemble E8 AI/ML AppKit or the Corstone-300 FVP.

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

## Generate and build

From the workspace root:

```sh
cbuild setup zephyr-m/zephyr-m.csolution.yml --active AppKit-E8 --packs
python model_conversion/create_ai_layer.py zephyr-m/zephyr-m.cbuild-mlops.yml
cbuild zephyr-m/zephyr-m.csolution.yml --active AppKit-E8
```

Use `Corstone-300-FVP` instead of `AppKit-E8` for the FVP target.

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
The application does not use the Ethos-U device, driver, `hal_ethos_u` module,
or Vela configuration on either target.

With GCC 15, the application keeps CMSIS-NN's `int32_t*` versus MVE intrinsic
`long*` compatibility diagnostic as a warning. Both types are 32-bit on this
target; other compiler diagnostics remain unchanged.

## Multiple-network example

The [TinyCNN multiple-network example](../model/TinyCNN/README.md) exports an
A16W8 `filter` and an INT8 `classify` into one `.pt2`, including runtime shape
constants. This solution includes a commented model input and an alternative
test selection. Follow the linked steps to generate it, select the test, and
compare the filter-then-classify output against a fixed reference.
