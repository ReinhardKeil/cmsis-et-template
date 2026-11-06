# Model report: TinyCNN

- input: `model/TinyCNN/model.py`
- input type: `Python model source`
- execution: `CPU with Ethos-U acceleration`
- parameters: `{"calibration_samples": 2, "input_shape": [1, 16, 16, 3]}`
- optimization: `default`
- CMSIS-NN required: `no`

## CPU fallback operators

- `quantized_decomposed::dequantize_per_tensor.out`
- `quantized_decomposed::quantize_per_tensor.out`

## Vela

- accelerator: `ethos-u85-256`
- system config: `Ethos_U85_SYS_DRAM_Mid`
- memory mode: `Shared_Sram`

```text
Network summary for out
Accelerator configuration               Ethos_U85_256
System configuration             Ethos_U85_SYS_DRAM_Mid
Memory mode                               Shared_Sram
Accelerator clock                                1000 MHz
Design peak SRAM bandwidth                      29.80 GB/s
Design peak DRAM bandwidth                      11.18 GB/s

Total SRAM used                                  5.25 KiB
Total DRAM used                                  5.25 KiB

CPU operators = 0 (0.0%)
NPU operators = 5 (100.0%)

Average SRAM bandwidth                           4.13 GB/s
Input   SRAM bandwidth                           0.01 MB/batch
Weight  SRAM bandwidth                           0.00 MB/batch
Output  SRAM bandwidth                           0.01 MB/batch
Total   SRAM bandwidth                           0.02 MB/batch
Total   SRAM bandwidth            per input      0.02 MB/inference (batch size 1)

Average DRAM bandwidth                           1.01 GB/s
Input   DRAM bandwidth                           0.00 MB/batch
Weight  DRAM bandwidth                           0.00 MB/batch
Output  DRAM bandwidth                           0.00 MB/batch
Total   DRAM bandwidth                           0.01 MB/batch
Total   DRAM bandwidth            per input      0.01 MB/inference (batch size 1)

Neural network macs                            134656 MACs/batch
```
