# AC6 issues

## Zephyr U55: floating-point `printf` causes a BusFault on AppKit-E8

Tested on 2026-11-04 with the Zephyr U55 Debug configuration on the Alif
AppKit-E8.

The AC6 build completes, programs the board, initializes the Ethos-U55, and
runs inference. It faults while formatting the first output value here:

```cpp
printf(" %.4f", static_cast<double>(data[i]));
```

Observed AC6 UART output and fault information:

```text
Output: 10 element(s): -0
***** BUS FAULT *****
Precise data bus error
BFAR Address: 0x39d61675
Faulting instruction address (r15/pc): 0x802019d6
Current thread: main
```

`CONFIG_REQUIRES_FLOAT_PRINTF=y` selects
`CONFIG_CBPRINTF_FP_SUPPORT=y`, and `%f` correctly receives a `double` because
of the explicit cast and the C/C++ variadic promotion rules.

### GCC comparison

The same application, model, AppKit-E8 board, and output loop were rebuilt with
GCC. Inference and floating-point output completed successfully:

```text
Output: 10 element(s): -0.0142 0.0206 0.0601 -0.0886 0.0791 0.0522 -0.0301 -0.0775 -0.2800 -0.0095
Test_result: PASS
```

This comparison isolates the failure to the AC6/Zephyr floating-point
`printf`/`cbprintf` path. It is not an Ethos-U inference or embedded-model
failure.

### Current workaround

Use GCC for the Zephyr U55 AppKit-E8 configuration, or avoid floating-point
format specifiers with AC6 (for example, print the raw `uint32_t` bit pattern)
until the AC6/Zephyr formatting incompatibility is resolved.
