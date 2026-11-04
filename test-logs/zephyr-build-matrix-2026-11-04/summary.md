# Zephyr GCC and AC6 build matrix

Date: 2026-11-04

## Scope

The `zephyr-u55` and `zephyr-u85` solutions were rebuilt for every combination
of:

- GCC 14.3.1 and Arm Compiler 6.24.0;
- Debug and Release;
- the Corstone FVP target and AppKit-E8 target.

Each build used `cbuild -r -c <context>` and has a complete log alongside this
report. These are compile/link tests of the FVP and connected-board target
configurations. The images were not flashed or executed on the FVP or AppKit.

## Environment

- CMSIS Solution extension: `arm.cmsis-csolution-1.72.1-79-20261103`
- CMSIS-Toolbox developer build: `cbuild 2.15.0-17-g2cc0cfd`
- Solution parser: `csolution 2.15.1+p45-gf04e3907`
- Zephyr: `4.5.0-rc3`, repository build `v4.5.0-rc3-11-gc4aa9324508f`
- west: `1.5.0`
- CMake: `4.4.2`
- Ninja: `1.12.1`
- GCC: `14.3.1`
- AC6: `6.24.0`
- ExecuTorch pack: `PyTorch::ExecuTorch@1.6.0-zephyrpreview1`

The released CMSIS-Toolbox 2.15.0 installed on `PATH` does not accept the
`$Pack(PyTorch::ExecuTorch)$` expression in `west-defs`. The newer CMSIS
Solution extension bundle was therefore used for every recorded build.

## Results

`Warnings` counts log lines containing `warning:`. Template-heavy diagnostics
can produce many warning lines from a single source construct.

| Solution | Compiler | Build | Target | Result | Warnings | Log |
|---|---|---|---|---:|---:|---|
| U55 | GCC | Debug | Corstone-300 FVP | PASS | 110 | [log](zephyr-u55-gcc-debug-corstone-300-fvp.log) |
| U55 | GCC | Release | Corstone-300 FVP | PASS | 109 | [log](zephyr-u55-gcc-release-corstone-300-fvp.log) |
| U55 | GCC | Debug | AppKit-E8 | PASS | 108 | [log](zephyr-u55-gcc-debug-appkit-e8.log) |
| U55 | GCC | Release | AppKit-E8 | PASS | 108 | [log](zephyr-u55-gcc-release-appkit-e8.log) |
| U85 | GCC | Debug | Corstone-320 FVP | PASS | 110 | [log](zephyr-u85-gcc-debug-corstone-320-fvp.log) |
| U85 | GCC | Release | Corstone-320 FVP | PASS | 109 | [log](zephyr-u85-gcc-release-corstone-320-fvp.log) |
| U85 | GCC | Debug | AppKit-E8 | PASS | 108 | [log](zephyr-u85-gcc-debug-appkit-e8.log) |
| U85 | GCC | Release | AppKit-E8 | PASS | 108 | [log](zephyr-u85-gcc-release-appkit-e8.log) |
| U55 | AC6 | Debug | Corstone-300 FVP | **FAIL** | 48 | [log](zephyr-u55-ac6-debug-corstone-300-fvp.log) |
| U55 | AC6 | Release | Corstone-300 FVP | **FAIL** | 48 | [log](zephyr-u55-ac6-release-corstone-300-fvp.log) |
| U55 | AC6 | Debug | AppKit-E8 | PASS | 48 | [log](zephyr-u55-ac6-debug-appkit-e8.log) |
| U55 | AC6 | Release | AppKit-E8 | PASS | 48 | [log](zephyr-u55-ac6-release-appkit-e8.log) |
| U85 | AC6 | Debug | Corstone-320 FVP | PASS | 49 | [log](zephyr-u85-ac6-debug-corstone-320-fvp.log) |
| U85 | AC6 | Release | Corstone-320 FVP | PASS | 49 | [log](zephyr-u85-ac6-release-corstone-320-fvp.log) |
| U85 | AC6 | Debug | AppKit-E8 | PASS | 48 | [log](zephyr-u85-ac6-debug-appkit-e8.log) |
| U85 | AC6 | Release | AppKit-E8 | PASS | 48 | [log](zephyr-u85-ac6-release-appkit-e8.log) |

Overall: **14 passed, 2 failed**.

## Blocking error

Both U55/Corstone-300 AC6 builds compile successfully but fail while linking
`zephyr_pre0.elf`:

```text
Error: L6225E: Number 0xffffffffffe80000... is too long.
```

The generated `linker_zephyr_pre0.cmd` contains the failing value at line 24,
column 29. Debug and Release fail identically. U55 AppKit-E8 and all U85 AC6
contexts link successfully, so this is specific to the Corstone-300 memory-map
generation for Arm linker.

## Warning summary

Warnings repeated across the builds include:

- Zephyr Kconfig: `UART_NS16550_ACCESS_WORD_ONLY` is requested as `y` but its
  `UART_NS16550` dependency is disabled.
- GCC/Zephyr: `fpscr` may be used uninitialized (`-Wmaybe-uninitialized`) in
  `arm-m-switch.c`.
- ExecuTorch: signed/unsigned comparisons (`-Wsign-compare`).
- ExecuTorch: float-to-double promotion (`-Wdouble-promotion`).
- ExecuTorch: deprecated `optional` alias (`-Wdeprecated-declarations`).
- AC6/Zephyr: ctype macro values used as booleans (`-Wint-in-bool-context`) in
  `fnmatch.c`.
- AC6/ExecuTorch: infinity with the enabled floating-point options
  (`-Wnan-infinity-disabled`).
- GCC linker: `zephyr_pre0.elf` and `zephyr.elf` have RWX load segments.
- Arm linker: duplicate `libethosu_core_driver.a` input is ignored (`L6304W`).

## Repository state after testing

Both Zephyr solution files now select:

```yaml
compiler: AC6
```

The GCC results were captured before changing those nodes to AC6.
