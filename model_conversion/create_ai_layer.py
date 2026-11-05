#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Create a CMSIS layer and optional Zephyr module from MLOps information.

    python model_conversion/create_ai_layer.py <solution>.cbuild-mlops.yml

Step 2 of the three-step flow:

    cbuild setup <solution>.csolution.yml --active <target>   # writes *.cbuild-mlops.yml
    python model_conversion/create_ai_layer.py <solution>.cbuild-mlops.yml  # this script
    cbuild <solution>.csolution.yml --active <target>         # compile and link

The *.cbuild-mlops.yml is what CMSIS-Toolbox generates from the `mlops:` node
of the csolution. This script reads the NPU and Vela settings from it, exports
the PyTorch model in model/model.py for that NPU (quantize, delegate to
Ethos-U, compile with Vela) and writes the complete AI layer into the
directory of the clayer named under `model.clayer`. When `model.zephyr` is
present, the script also writes Zephyr integration files. The other supported
keys of the `model:` node (input-shape, calibration-samples) are model parameters that
CMSIS-Toolbox passes through to this script:

    ai_layer.clayer.yml   CMSIS layer metadata
    CMakeLists.txt and
    *.conf                optional Zephyr module metadata
    model_pte.c / .h      the ExecuTorch program as a C array
    model.pte             the program itself, for inspection
    VELA_SUMMARY.md       effective Vela settings and textual report

The script runs itself in the workspace's .venv (see setup_venv.py). Paths in
the MLOps file are resolved from the solution directory containing that file.
"""

from __future__ import annotations

import io
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from pathlib import Path

PACK = "PyTorch::ExecuTorch"
SYMBOL = "model_pte"
WORKSPACE_DIR = Path(__file__).resolve().parent.parent


def run_in_venv(workspace_dir: Path) -> None:
    """Re-run under the workspace's .venv unless already running in it.

    Deciding by "does torch import" is not enough: a torch installed for the
    host interpreter would keep the export outside the environment with the
    pinned executorch. sys.prefix is the venv directory when running inside it
    (comparing interpreter paths does not work: venv symlinks resolve to the
    base interpreter).
    """
    venv = workspace_dir / ".venv"
    if Path(sys.prefix).resolve() == venv.resolve():
        return
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        sys.exit(
            f"{venv} does not exist.\n"
            "Create it first with model_conversion/setup_venv.sh (Linux/macOS) "
            "or model_conversion/setup_venv.bat (Windows)"
        )
    print(f"[ai_layer] running in {python}", flush=True)
    script = Path(__file__).resolve()
    sys.exit(subprocess.run([str(python), str(script), *sys.argv[1:]]).returncode)


def pack_root() -> Path:
    """The CMSIS pack root: $CMSIS_PACK_ROOT, else cpackget's default."""
    if env := os.environ.get("CMSIS_PACK_ROOT"):
        return Path(env)
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "Arm/Packs"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "arm/packs"


def executorch_version(mlops_file: Path) -> str:
    """The PyTorch::ExecuTorch pack version cbuild setup resolved, from <solution>.cbuild-pack.yml.

    The file is a lock file that keeps earlier resolutions, so an unversioned
    selector can still point at an older pack; the entry selected by the
    csolution's exact pin (PyTorch::ExecuTorch@<version>) is the one in use.
    """
    import yaml

    pack_file = mlops_file.with_name(mlops_file.name.replace(".cbuild-mlops.yml", ".cbuild-pack.yml"))
    fallback = None
    for entry in yaml.safe_load(pack_file.read_text())["cbuild-pack"]["resolved-packs"]:
        name, _, version = entry["resolved-pack"].partition("@")
        selectors = entry.get("selected-by-pack", [])
        if name != PACK or not selectors:
            continue
        if f"{PACK}@{version}" in selectors:
            return version
        fallback = fallback or version
    if fallback:
        return fallback
    sys.exit(f"{pack_file}: {PACK} is not among the resolved packs")


def executorch_pack(version: str) -> Path:
    """Directory of the installed PyTorch::ExecuTorch pack of that version."""
    vendor, _, pack = PACK.partition("::")
    return pack_root() / vendor / pack / version


# Vela options that are arguments of EthosUCompileSpec, and the ones the export
# sets itself: the configuration file comes from vela.ini of the cbuild-mlops.yml.
SPEC_OPTIONS = ("accelerator-config", "system-config", "memory-mode")
EXPORT_OPTIONS = ("config", "output-format", "output-dir")


def vela_options(options: str) -> tuple[dict[str, str], list[str]]:
    """Split vela.options of the cbuild-mlops.yml: the EthosUCompileSpec arguments, and every other option.

    The other options are the `misc:` of the csolution's mlops: node (for
    example `--optimise Size`); they reach Vela as extra flags, written as
    `--name=value`.
    """
    tokens = shlex.split(options)
    spec, extra = {}, []
    while tokens:
        token = tokens.pop(0)
        if not token.startswith("--"):
            sys.exit(f"vela options: unexpected '{token}' in '{options}'")
        name, has_value, value = token[2:].partition("=")
        if not has_value and tokens and not tokens[0].startswith("-"):
            value = tokens.pop(0)
        if name in EXPORT_OPTIONS:
            sys.exit(
                f"vela options: --{name} is set by the export; "
                "the configuration file is vela: ini: of the mlops: node"
            )
        if name in SPEC_OPTIONS:
            if name in spec:
                sys.exit(f"vela options: --{name} is given twice in '{options}'")
            spec[name] = value
        else:
            extra.append(f"--{name}={value}" if value else f"--{name}")
    return spec, extra


def compile_spec(mlops: dict, mlops_dir: Path):
    """EthosUCompileSpec for an NPU target; None selects Cortex-M generation."""
    if mlops["model"].get("cpu"):
        return None
    npu = mlops.get("npu")
    if not npu:
        return None

    from executorch.backends.arm.ethosu import EthosUCompileSpec
    vela = mlops.get("vela", {})
    spec, extra = vela_options(vela.get("options", ""))

    target = spec.get("accelerator-config") or f"{npu['type'].lower()}-{npu.get('macs', 256)}"
    kwargs = {
        "target": target,
        "system_config": spec.get("system-config"),
        "memory_mode": spec.get("memory-mode"),
        "extra_flags": extra,
    }
    if vela.get("ini"):
        # ExecuTorch stores the path in the compile spec, and the spec ends up
        # in the .pte. A path relative to the working directory keeps the
        # program identical between checkouts; Vela resolves it from there.
        kwargs["config_ini"] = os.path.relpath(mlops_dir / vela["ini"])
    print(f"[ai_layer] Vela: {kwargs}")
    return EthosUCompileSpec(**kwargs)


def cpu_target(mlops: dict):
    """Create the Cortex-M backend target from model.cpu or processor.type."""
    from executorch.backends.cortex_m.target_config import CortexMTargetConfig

    core = mlops["model"].get("cpu") or mlops.get("processor", {}).get("type")
    if not core:
        sys.exit("the cbuild-mlops.yml names neither an NPU nor a processor")
    target = core.lower().replace("+", "plus")
    try:
        config = CortexMTargetConfig.from_target_string(target)
    except ValueError as error:
        sys.exit(f"no Cortex-M backend target for processor {core!r}: {error}")
    print(f"[ai_layer] Cortex-M: {config.target_string}, CMSIS-NN {config.backend.name} kernels")
    return config


def model_params(mlops: dict) -> dict:
    """The extra keys of the model: node, as keyword arguments for model/model.py."""
    params = {}
    for key, value in mlops["model"].items():
        if key == "input-shape":
            params["input_shape"] = tuple(int(d) for d in re.split(r"[x,]", str(value)))
        elif key == "calibration-samples":
            params["calibration_samples"] = int(value)
        elif key not in ("clayer", "name", "zephyr", "cpu"):
            print(f"[ai_layer] warning: ignoring unknown model key {key!r}", file=sys.stderr)
    if params:
        print(f"[ai_layer] model parameters: {params}")
    return params


def load_model(params: dict):
    """Load the shared model and return its exported graph and calibration data."""
    import torch

    # Model sources are shared by all solution directories at workspace level.
    sys.path.insert(0, str(WORKSPACE_DIR / "model"))
    from model import get_calibration_inputs, get_model

    model = get_model(**{k: v for k, v in params.items() if k == "input_shape"})
    samples = get_calibration_inputs(**params)
    example = (samples[0],)
    graph = torch.export.export(model, example).module()
    strip_guards_fn(graph)
    return graph, samples, example


def strip_guards_fn(graph_module) -> None:
    """Remove the dead guard module emitted by recent Torch export versions."""
    changed = False
    for node in list(graph_module.graph.nodes):
        if node.op == "call_module" and str(node.target) == "_guards_fn":
            graph_module.graph.erase_node(node)
            changed = True
    if changed:
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()


def quantize(graph, quantizer, samples):
    """Annotate, calibrate with every sample, and convert a PT2E graph."""
    import torch
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    prepared = prepare_pt2e(graph, quantizer)
    with torch.no_grad():
        for sample in samples:
            prepared(sample)
    return convert_pt2e(prepared)


def export_model(spec, params: dict, solution_dir: Path) -> bytes:
    """Quantize model/model.py, delegate it to the Ethos-U and return the .pte."""
    import torch
    from executorch.backends.arm.ethosu import EthosUPartitioner
    from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower

    graph, samples, example = load_model(params)

    # Quantize the whole graph so the partitioner can move every node into the
    # Ethos-U delegate; only a float<->int8 boundary stays on the CPU.
    quantizer = EthosUQuantizer(spec)
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    quantized = quantize(graph, quantizer, samples)

    edge = to_edge_transform_and_lower(
        torch.export.export(quantized, example),
        partitioner=[EthosUPartitioner(spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    program = edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
    return bytes(program.buffer)


def export_model_cpu(target, params: dict, solution_dir: Path) -> bytes:
    """Quantize and lower model/model.py for Cortex-M with CMSIS-NN kernels."""
    import torch
    from executorch.backends.cortex_m.edge_compile_config import cortex_m_edge_compile_config
    from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
    from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
    from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower

    graph, samples, example = load_model(params)
    quantized = quantize(graph, CortexMQuantizer(use_explicit_layout=True), samples)
    exported = torch.export.export(quantized, example)
    strip_guards_fn(exported.graph_module)
    edge = to_edge_transform_and_lower(exported, compile_config=cortex_m_edge_compile_config())
    edge._edge_programs["forward"] = CortexMPassManager(
        edge.exported_program(), target_config=target, use_explicit_layout=True
    ).transform()
    program = edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
    return bytes(program.buffer)


class Tee(io.TextIOBase):
    """Copy text to the terminal while retaining it for VELA_SUMMARY.md."""

    def __init__(self, terminal, capture: io.StringIO):
        self.terminal = terminal
        self.capture = capture

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.capture.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()


def compiler_flag_value(flags: list[str], option: str) -> str:
    """Return an effective Vela option from EthosUCompileSpec.compiler_flags."""
    prefix = f"--{option}="
    return next((flag[len(prefix) :] for flag in flags if flag.startswith(prefix)), "not specified")


def vela_summary(model_name: str, spec, report: str) -> str:
    """Markdown record of the effective Vela configuration and textual report."""
    report = report.replace("\r\n", "\n").replace("\r", "\n").strip()
    return (
        f"# Vela report: {model_name}\n\n"
        f"- accelerator: `{spec.target}`\n"
        f"- system config: `{compiler_flag_value(spec.compiler_flags, 'system-config')}`\n"
        f"- memory mode: `{compiler_flag_value(spec.compiler_flags, 'memory-mode')}`\n\n"
        f"```text\n{report}\n```\n"
    )


def replace_outputs(layer_dir: Path, outputs: dict[str, bytes | str]) -> None:
    """Stage a complete AI layer, then atomically replace destination files."""
    layer_dir.mkdir(parents=True, exist_ok=True)
    # tempfile.TemporaryDirectory deliberately creates a private directory on
    # Windows. Files moved out of it keep that restrictive ACL and cannot be
    # read by build tools running as the interactive user. A normal child
    # directory inherits the destination ACL while still allowing all outputs
    # to be staged before any destination is replaced.
    stage = layer_dir / f".generate-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        for name, contents in outputs.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(contents, bytes):
                path.write_bytes(contents)
            elif isinstance(contents, str):
                path.write_text(contents, encoding="utf-8", newline="\n")
            else:
                raise TypeError(f"unsupported generated output type for {name}")

        for name in outputs:
            destination = layer_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / name, destination)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


FAMILY = {
    "aten": "Portable",
    "dim_order_ops": "Portable",
    "quantized_decomposed": "Quantized",
    "cortex_m": "Cortex-M",
}
ALIASES = {
    "quantized_depthwise_conv2d": "quantized_dw_conv2d",
    "quantized_transpose_conv2d": "quantized_tr_conv2d",
}


def pack_components(pack: Path) -> tuple[set[str], set[str]]:
    """Return operator components and the subset whose conditions require CMSIS-NN."""
    root = ET.parse(next(pack.glob("*.pdsc"))).getroot()
    requires = {
        condition.get("id"): [
            requirement.get("condition")
            for requirement in condition
            if requirement.tag == "require" and requirement.get("condition")
        ]
        for condition in root.iter("condition")
    }

    def needs_cmsis_nn(condition: str | None) -> bool:
        return condition == "CMSIS-NN" or any(
            needs_cmsis_nn(requirement) for requirement in requires.get(condition, [])
        )

    operators = {
        component.get("Csub"): component.get("condition")
        for component in root.iter("component")
        if component.get("Cgroup") == "ExecuTorch Operators"
    }
    return set(operators), {
        name for name, condition in operators.items() if needs_cmsis_nn(condition)
    }


def components(pte: bytes, pack: Path, *, delegated: bool) -> tuple[list[str], list[str], bool]:
    """Select runtime/operators and report whether their conditions require CMSIS-NN.

    The pack's "Extension Tensor" is not selected: its tensor_ptr_maker.cpp
    needs std::random_device, which the LLVM embedded toolchain lacks; the
    runner wraps its input in a TensorImpl instead.
    """
    available, cmsis_nn = pack_components(pack)
    namespaces = "|".join(FAMILY)

    selected, unknown = set(), []
    for ns, op in sorted(set(re.findall(rf"({namespaces})::(\w+)".encode(), pte))):
        ns, op = ns.decode(), op.decode()
        op = ALIASES.get(op, op)
        candidates = [
            f"{FAMILY[ns]} {op}",
            f"{FAMILY[ns]} {op.lstrip('_')}",
            f"{FAMILY[ns]} {re.sub(r'_(per_tensor|per_channel|byte|copy|nhwc)$', '', op)}",
        ]
        if match := next((c for c in candidates if c in available), None):
            selected.add(match)
        else:
            unknown.append(f"{ns}::{op}")
    if unknown:
        print(f"[ai_layer] warning: no component in {pack.name} for {unknown}", file=sys.stderr)

    runtime = ["Runtime", "Kernel Utils", "Kernel Registration"]
    if delegated:
        runtime.append("Backend EthosU")
    return runtime, sorted(selected), bool(selected & cmsis_nn)


def c_array(pte: bytes) -> str:
    rows = [", ".join(f"0x{b:02x}" for b in pte[i : i + 16]) for i in range(0, len(pte), 16)]
    return (
        "// Generated by create_ai_layer.py -- do not edit.\n"
        f"__attribute__((aligned(16))) const unsigned char {SYMBOL}[] = {{\n  "
        + ",\n  ".join(rows)
        + f"\n}};\nconst unsigned long {SYMBOL}_size = sizeof({SYMBOL});\n"
    )


HEADER = f"""// Generated by create_ai_layer.py -- do not edit.
#pragma once
#ifdef __cplusplus
extern "C" {{
#endif
extern const unsigned char {SYMBOL}[];
extern const unsigned long {SYMBOL}_size;
#ifdef __cplusplus
}}
#endif
"""


def clayer(
    mlops: dict,
    runtime: list[str],
    operators: list[str],
    cmsis_nn: bool,
    mlops_file: Path,
    version: str,
) -> str:
    try:
        mlops_path = os.path.relpath(mlops_file, WORKSPACE_DIR)
    except ValueError:
        # Windows cannot construct a relative path across drive letters. This
        # path is only used in the regeneration hint, so keep it absolute.
        mlops_path = mlops_file
    lines = [
        f"# Generated by create_ai_layer.py from {mlops_file.name} -- do not edit.",
        f"# Re-run `python model_conversion/create_ai_layer.py {mlops_path}` after changing",
        "# model/model.py or the csolution's mlops: node.",
        "layer:",
        "  type: AI",
        f"  description: {mlops.get('description', mlops['model'].get('name', 'AI layer'))}",
        "",
        "  packs:",
        f"    - pack: {PACK}@{version}",
        *(["    - pack: ARM::CMSIS-NN"] if cmsis_nn else []),
        "",
        "  define:",
        "    - ET_LOG_ENABLED: 0",
        *(["    - ARM_NN_ENABLE_F32: 1", "    - ARM_NN_ENABLE_F16: 1"] if cmsis_nn else []),
        "",
        "  add-path:",
        "    - .",
        "",
        "  components:",
        *[f"    - component: Machine Learning:ExecuTorch:{c}" for c in runtime],
        *[f"    - component: Machine Learning:ExecuTorch Operators:{c}" for c in operators],
        *(["    - component: CMSIS:NN Lib"] if cmsis_nn else []),
        "",
        "  groups:",
        f"    - group: {mlops['model'].get('name', 'Model')}",
        "      files:",
        f"        - file: ./{SYMBOL}.c",
        f"        - file: ./{SYMBOL}.h",
        "",
    ]
    return "\n".join(lines)


ZEPHYR_CMAKE = '''# Generated by create_ai_layer.py -- do not edit.
# Include this directory after find_package(Zephyr) creates the app target.
target_sources(app PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/model_pte.c")
target_include_directories(app PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
'''

ZEPHYR_COMPONENTS = '''# Generated by create_ai_layer.py -- do not edit.
CONFIG_CMSIS_EXECUTORCH=y

CONFIG_CMSIS_EXECUTORCH_RUNTIME=y
CONFIG_CMSIS_EXECUTORCH_KERNEL_UTILS=y
CONFIG_CMSIS_EXECUTORCH_KERNEL_REGISTRATION=y
'''


def zephyr_metadata(operators: list[str], pack: Path, delegated: bool) -> dict[str, str]:
    """Generate the Zephyr module files for the selected operator components."""
    symbols = [re.sub(r"[^A-Z0-9]+", "_", operator.upper()).strip("_") for operator in operators]
    kconfig = (pack / "zephyr" / "Kconfig").read_text()
    available = set(re.findall(r"(?m)^config CMSIS_EXECUTORCH_OP_([A-Z0-9_]+)\s*$", kconfig))
    missing = sorted(set(symbols) - available)
    if missing:
        raise ValueError(f"pack Kconfig options missing for {missing}")
    operator_config = "# Generated by create_ai_layer.py -- do not edit.\n" + "".join(
        f"CONFIG_CMSIS_EXECUTORCH_OP_{symbol}=y\n" for symbol in sorted(set(symbols))
    )
    components = ZEPHYR_COMPONENTS
    if delegated:
        components += "CONFIG_CMSIS_EXECUTORCH_BACKEND_ETHOS_U=y\n"
    return {
        "CMakeLists.txt": ZEPHYR_CMAKE,
        "components.conf": components,
        "operators.conf": operator_config,
    }


def main() -> None:
    if len(sys.argv) != 2 or not sys.argv[1].endswith(".cbuild-mlops.yml"):
        sys.exit(f"usage: {Path(__file__).name} <solution>.cbuild-mlops.yml")

    # CMSIS-Toolbox paths are relative to the generated MLOps file. The Python
    # environment is workspace-wide, while model sources and generated files
    # belong to the selected solution.
    mlops_file = Path(sys.argv[1]).resolve()
    solution_dir = mlops_file.parent
    run_in_venv(WORKSPACE_DIR)

    import yaml

    mlops = yaml.safe_load(mlops_file.read_text())["cbuild-mlops"]
    model = mlops["model"]
    if "zephyr" in model:
        # YAML loads an empty ``zephyr:`` node as None. Treat it as the empty
        # mapping so the concise form and ``zephyr: {}`` have identical meaning.
        if model["zephyr"] is None:
            model["zephyr"] = {}
        elif not isinstance(model["zephyr"], dict):
            sys.exit("model.zephyr must be empty or a mapping, for example: zephyr:")
    layer_file = solution_dir / model["clayer"]
    layer_dir = layer_file.parent

    params = model_params(mlops)
    spec = compile_spec(mlops, solution_dir)
    vela_report = None
    if spec is None:
        pte = export_model_cpu(cpu_target(mlops), params, solution_dir)
    else:
        vela_report = io.StringIO()
        with redirect_stdout(Tee(sys.stdout, vela_report)):
            pte = export_model(spec, params, solution_dir)
    version = executorch_version(mlops_file)
    runtime, operators, cmsis_nn = components(
        pte, executorch_pack(version), delegated=spec is not None
    )

    outputs: dict[str, bytes | str] = {
        "model.pte": pte,
        f"{SYMBOL}.c": c_array(pte),
        f"{SYMBOL}.h": HEADER,
        layer_file.name: clayer(
            mlops, runtime, operators, cmsis_nn, mlops_file, version
        ),
    }
    if vela_report is not None:
        outputs["VELA_SUMMARY.md"] = vela_summary(
            model.get("name", "Model"), spec, vela_report.getvalue()
        )
    if "zephyr" in model:
        outputs.update(
            zephyr_metadata(operators, executorch_pack(version), delegated=spec is not None)
        )
    replace_outputs(layer_dir, outputs)

    print(f"[ai_layer] {len(pte)} byte program, operators: {operators}")
    print(f"[ai_layer] wrote {layer_file}")
    if vela_report is not None:
        print(f"[ai_layer] wrote {layer_dir / 'VELA_SUMMARY.md'}")
    if "zephyr" in model:
        print(f"[ai_layer] wrote {layer_dir / 'CMakeLists.txt'}")


if __name__ == "__main__":
    main()
