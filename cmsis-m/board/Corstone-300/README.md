# Board: Arm V2M-MPS3-SSE-300-FVP

This local board layer supplies the Cortex-M55 startup, linker memory regions,
and CMSIS UART configuration for the CPU-only solution. It uses the
`ARM::V2M_MPS3_SSE_300_BSP` pack for the device and board drivers.

`Board.clayer.yml` includes `main.c` and `main.h`. The `RTE/` directory contains
the editable device and CMSIS-Compiler configuration files. UART0 provides
standard input, output, and error.

`fvp_config.txt` configures the CPU-only Corstone-300 FVP and exits on UART EOT.
