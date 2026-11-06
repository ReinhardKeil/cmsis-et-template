# Model directory migration test summary

Date: 2026-11-06
Host: Windows, CMSIS-Toolbox 2.15-17-g2cc0cfd, Zephyr 4.5.0-rc3
Board: Alif Ensemble AppKit-E8, M55_HP

## Target verification

| Example | Target | Conversion | Build | Runtime result |
|---|---|---:|---:|---|
| `cmsis-m` | SSE-300-M55 FVP | PASS | PASS | FVP load/run completed |
| `cmsis-u55` | SSE-300-U55 FVP | PASS | PASS | FVP load/run completed |
| `cmsis-u85` | SSE-320-U85 FVP | PASS | PASS | FVP load/run completed |
| `zephyr-m` | AppKit-E8 | PASS | PASS | PASS — debugger reached `Test_result: PASS` |
| `zephyr-u55` | Corstone-300 FVP | PASS | PASS | FVP load/run completed |
| `zephyr-u55` | AppKit-E8 | PASS | PASS | PASS — debugger reached `Test_result: PASS` |
| `zephyr-u85` | Corstone-320 FVP | PASS | PASS | PASS — FVP console printed `Test_result: PASS` |
| `zephyr-u85` | AppKit-E8 | PASS | PASS | PASS — debugger reached `Test_result: PASS` |

`zephyr-m` had no FVP target. The CMSIS action API reported successful FVP
completion for the other rows but did not return their UART stream; therefore
those rows deliberately say "load/run completed" rather than claiming captured
PASS text.

## Environment notes

- The CMSIS examples use ExecuTorch 1.5.1; the Zephyr preview pack uses the
  1.6.0 Python exporter. The local virtual environment was temporarily changed
  for Zephyr generation and restored to the repository's 1.5.1 requirements.
