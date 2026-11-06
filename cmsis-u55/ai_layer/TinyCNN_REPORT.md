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

- accelerator: `ethos-u55-128`
- system config: `Ethos_U55_High_End_Embedded`
- memory mode: `Shared_Sram`

```text
Network summary for out
Accelerator configuration               Ethos_U55_128
System configuration             Ethos_U55_High_End_Embedded
Memory mode                               Shared_Sram
Accelerator clock                                 500 MHz
Design peak SRAM bandwidth                       3.73 GB/s
Design peak Off-chip Flash bandwidth             0.47 GB/s

Total SRAM used                                  6.48 KiB
Total Off-chip Flash used                        4.98 KiB

CPU operators = 0 (0.0%)
NPU operators = 7 (100.0%)

Average SRAM bandwidth                           1.03 GB/s
Input   SRAM bandwidth                           0.01 MB/batch
Weight  SRAM bandwidth                           0.00 MB/batch
Output  SRAM bandwidth                           0.01 MB/batch
Total   SRAM bandwidth                           0.03 MB/batch
Total   SRAM bandwidth            per input      0.03 MB/inference (batch size 1)

Average Off-chip Flash bandwidth                 0.20 GB/s
Input   Off-chip Flash bandwidth                 0.00 MB/batch
Weight  Off-chip Flash bandwidth                 0.00 MB/batch
Output  Off-chip Flash bandwidth                 0.00 MB/batch
Total   Off-chip Flash bandwidth                 0.00 MB/batch
Total   Off-chip Flash bandwidth  per input      0.00 MB/inference (batch size 1)

Neural network macs                            134656 MACs/batch
```
