# CMSIS template AppKit-E8 and FVP test

Date: 2026-11-09

## Environment

- VS Code CMSIS Solution extension: `arm.cmsis-csolution-1.72.1-86-20261109`
- Bundled `csolution`: `2.15.1+p48-g9666cdf1`
- Bundled `cbuild`: `2.15.0-17-g2cc0cfd`
- Compiler: Arm Compiler for Embedded `6.24`
- AppKit pack: `AlifSemiconductor::Ensemble@2.2.2-dev40`
- AppKit board: `AppKit-E8-AIML`, device `AE822FA0E5597BS0:M55_HE`
- Probe/debug: J-Link Server, SWD at 4 MHz
- Serial: COM4, 115200 baud

Only the CMSIS-Toolbox bundled with the VS Code extension was used.

## Applied AppKit board layers

| Template | Target | Layer |
|---|---|---|
| `cmsis-m` | `AppKit-E8-M55` | `Boards/AppKit-e8/Layers/M55_HE/Board_HE.clayer.yml` |
| `cmsis-u55` | `AppKit-E8-U55` | `Boards/AppKit-e8/Layers/M55_HE/Board_HE.clayer.yml` |
| `cmsis-u85` | `AppKit-E8-U85` | `Boards/AppKit-e8/Layers/M55_HE/Board_HE-U85.clayer.yml` |

The layers consume CMSIS-RTOS2, so the AppKit contexts include
`ARM::CMSIS-RTX@5.9.1`, `CMSIS:RTOS2:Keil RTX5&Source`, and
`CMSIS:OS Tick:SysTick`. The AppKit allocator pools are reduced from 4 MiB to
3 MiB each so the images fit the layer's 8 MiB `RW_SRAM` region. FVP contexts
retain the original pool sizes.

## Results

| Template | AppKit build | AppKit execution | FVP rebuild | FVP execution |
|---|---:|---:|---:|---:|
| `cmsis-m` | PASS | FAIL | PASS | INCONCLUSIVE |
| `cmsis-u55` | PASS | FAIL | PASS | PASS |
| `cmsis-u85` | PASS | NOT RUN | PASS | PASS |

### AppKit execution

`cmsis-u55` and `cmsis-m` were programmed and launched with the explicit
`M55_HE JLink` target set. In both cases the J-Link session connected and the
target ran, but COM4 produced no output for 30 seconds and `app_main` was not
reached. Therefore these are runtime failures, not passes. The U85 image was
not programmed after the common failure was reproduced on the other two
images.

### FVP execution

The U55 and U85 `load_and_run` jobs completed normally after clean rebuilds.
The M55 FVP terminated with exit code 0 after two seconds; this is consistent
with the application's EOT shutdown after `Test_result: PASS`, but the CMSIS
Developer Assistant classified the early process exit as a failed persistent
run and did not expose the FVP stdout. Its result is therefore recorded as
inconclusive rather than pass.

## Build issues found and resolved

1. `$Pack(AlifSemiconductor::Ensemble)$` was not expanded when used as a
   solution variable containing a layer path. The test targets use the exact
   supplied pack path.
2. The AppKit layer's `ethosu_os_rtos2.c` includes `cmsis_os2.h`
   unconditionally; adding RTX5 and OS Tick resolved the dependency.
3. Default 4 MiB method and temporary pools overflowed `RW_SRAM` by 63,380
   bytes (`L6220E`). AppKit-only 3 MiB pools resolved the link failure.
4. Explicit AppKit target sets were required to prevent the extension from
   reusing the previously selected FVP launch configuration.

## Overall

- Builds: **6/6 passed**.
- Confirmed runtime passes: **2/6** (U55 FVP and U85 FVP).
- Runtime failures: **2/6** (M55 AppKit and U55 AppKit: no UART output and no
  entry into `app_main`).
- Not run/inconclusive: **2/6** (U85 AppKit not programmed; M55 FVP stdout not
  observable through the test interface).
