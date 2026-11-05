# CMSIS Cortex-M CPU-only quick start

This example runs the TinyCNN model entirely on the Cortex-M55 of the
Corstone-300 FVP. It uses the ExecuTorch Cortex-M backend and CMSIS-NN, with no
Ethos-U NPU or Vela compilation.

Complete the workspace-level [prerequisites](../README.md#prerequisites) first.
Commands below run from the workspace root.

## Generate, build, and run

```console
cbuild setup cmsis-m/cmsis-m.csolution.yml --active SSE-300-M55 --packs
python model_conversion/create_ai_layer.py cmsis-m/cmsis-m.cbuild-mlops.yml
cbuild cmsis-m/cmsis-m.csolution.yml --active SSE-300-M55
```

The solution intentionally has no `mlops.npu` node. The generator obtains
`Cortex-M55` from the generated `processor:` node, applies the Cortex-M
quantizer and lowering passes, and selects the required Cortex-M operator
components. If those operators require CMSIS-NN, the generated layer adds the
unversioned `ARM::CMSIS-NN` pack and `CMSIS:NN Lib` component.

Run the image on the CPU-only FVP:

```console
FVP_Corstone_SSE-300 \
  -f cmsis-m/board/Corstone-300/fvp_config.txt --simlimit 60 \
  -a cmsis-m/out/cmsis-m/SSE-300-M55/Debug/cmsis-m.elf
```

The FVP configuration contains no Ethos-U settings. A successful run ends
with:

```text
Test_result: PASS
```

The generated AI layer is committed so the example can be built immediately.
Re-run setup and model conversion after changing the model, target, or pack
version.
