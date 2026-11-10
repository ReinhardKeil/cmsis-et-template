# Input report: TinyCNN

- input: `model/TinyCNN/model.py`
- input type: `.py`
- execution: `CPU with Ethos-U acceleration`
- parameters: `{"calibration_samples": 2, "input_shape": [1, 16, 16, 3]}`
- optimization: `default`
- CMSIS-NN required: `no`
- methods: `forward`

## CPU fallback operators

- `quantized_decomposed::dequantize_per_tensor.out`
- `quantized_decomposed::quantize_per_tensor.out`

## Vela

- accelerator: `ethos-u55-256`
- system config: `RTSS_HP_SRAM_MRAM`
- memory mode: `Shared_Sram`

```text
Network summary for out
Accelerator configuration               Ethos_U55_256
System configuration                RTSS_HP_SRAM_MRAM
Memory mode                               Shared_Sram
Accelerator clock                                 400 MHz
Design peak SRAM bandwidth                       2.27 GB/s
Design peak Off-chip Flash bandwidth             0.36 GB/s

Total SRAM used                                  5.00 KiB
Total Off-chip Flash used                        4.98 KiB

CPU operators = 0 (0.0%)
NPU operators = 7 (100.0%)

Average SRAM bandwidth                           1.42 GB/s
Input   SRAM bandwidth                           0.01 MB/batch
Weight  SRAM bandwidth                           0.00 MB/batch
Output  SRAM bandwidth                           0.01 MB/batch
Total   SRAM bandwidth                           0.02 MB/batch
Total   SRAM bandwidth            per input      0.02 MB/inference (batch size 1)

Average Off-chip Flash bandwidth                 0.37 GB/s
Input   Off-chip Flash bandwidth                 0.00 MB/batch
Weight  Off-chip Flash bandwidth                 0.00 MB/batch
Output  Off-chip Flash bandwidth                 0.00 MB/batch
Total   Off-chip Flash bandwidth                 0.00 MB/batch
Total   Off-chip Flash bandwidth  per input      0.00 MB/inference (batch size 1)

Neural network macs                            134656 MACs/batch
```
