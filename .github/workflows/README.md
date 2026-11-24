# GitHub automation

The default workflows test the user-facing `TinyCNN.py` examples:

- `test-cmsis-m.yml`, `test-cmsis-u55.yml`, and `test-cmsis-u85.yml` build the
  CMSIS examples with AC6, GCC, and Clang.
- `test-zephyr-gcc.yml` builds all Zephyr examples with GCC.
- `test-python-environment.yml` checks the Python model-conversion environment.

The GCC-only variant workflows cover the pre-exported models:

- `cmsis-variants.yml` lists the CMSIS variants in a constant matrix.
- `zephyr-variants.yml` lists the Zephyr variants in a constant matrix.

Files in `variants/` are CI-only solutions for `TinyCNN.pt2` and
`TinyCNNMultiple.pt2`. A workflow copies each variant beside its corresponding
user-facing solution so relative paths and `$SolutionDir()$` resolve normally.
Add new variant files to the corresponding workflow matrix.

`vcpkg-zephyr.json` selects the tools used by the Zephyr workflows.
