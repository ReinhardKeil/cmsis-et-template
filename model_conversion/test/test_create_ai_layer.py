#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Tests for model_conversion/create_ai_layer.py."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "model_conversion" / "create_ai_layer.py"
SPEC = importlib.util.spec_from_file_location("create_ai_layer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generator
SPEC.loader.exec_module(generator)


class CreateAiLayerTests(unittest.TestCase):
    @staticmethod
    def compile_spec() -> mock.Mock:
        return mock.Mock(
            target="ethos-u55-256",
            compiler_flags=[
                "--accelerator-config=ethos-u55-256",
                "--system-config=Test_System",
                "--memory-mode=Shared_Sram",
            ],
        )

    @staticmethod
    def inspection(*methods: str, delegated: bool = False) -> generator.PteInspection:
        return generator.PteInspection(
            methods or ("forward",),
            ("EthosUBackend",) if delegated else (),
            ("ethos-u55-256",) if delegated else (),
            ("Test_System",) if delegated else (),
            ("Shared_Sram",) if delegated else (),
        )

    def test_workspace_is_parent_of_model_conversion_directory(self) -> None:
        self.assertEqual(generator.WORKSPACE_DIR, ROOT)

    def test_run_in_venv_uses_workspace_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = root / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            python.parent.mkdir(parents=True)
            python.touch()
            completed = mock.Mock(returncode=7)
            with mock.patch.object(generator.sys, "prefix", str(root / "host")), mock.patch.object(
                generator.subprocess, "run", return_value=completed
            ) as run, self.assertRaises(SystemExit) as exit_context:
                generator.run_in_venv(root)

            self.assertEqual(exit_context.exception.code, 7)
            self.assertEqual(Path(run.call_args.args[0][0]), python)

    def test_run_in_venv_does_not_restart_inside_workspace_venv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(generator.sys, "prefix", str(root / ".venv")), mock.patch.object(
                generator.subprocess, "run"
            ) as run:
                generator.run_in_venv(root)
            run.assert_not_called()

    def test_vela_options_separate_compile_spec_and_extra_flags(self) -> None:
        spec, extra = generator.vela_options(
            "--accelerator-config=ethos-u55-128 "
            "--system-config Test_System --memory-mode=Shared_Sram "
            "--optimise Size --verbose-all"
        )
        self.assertEqual(
            spec,
            {
                "accelerator-config": "ethos-u55-128",
                "system-config": "Test_System",
                "memory-mode": "Shared_Sram",
            },
        )
        self.assertEqual(extra, ["--optimise=Size", "--verbose-all"])

    def test_vela_options_reject_export_owned_and_duplicate_options(self) -> None:
        with self.assertRaisesRegex(SystemExit, "set by the export"):
            generator.vela_options("--output-dir generated")
        with self.assertRaisesRegex(SystemExit, "given twice"):
            generator.vela_options("--memory-mode one --memory-mode two")

    def test_model_params_converts_supported_values(self) -> None:
        mlops = {
            "model": {
                "clayer": "ai_layer/ai_layer.clayer.yml",
                "name": "TinyCNN",
                "input-shape": "1x16,16x3",
                "calibration-samples": "4",
            }
        }
        self.assertEqual(
            generator.model_params(mlops),
            {"input_shape": (1, 16, 16, 3), "calibration_samples": 4},
        )

    def test_model_inputs_validate_multi_model_form(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "npu.py").write_text("# model", encoding="utf-8")
            (root / "cpu.pte").write_bytes(b"PTE")
            mlops = {
                "processor": {"type": "Cortex-M55"},
                "npu": {"type": "Ethos-U55"},
                "model": {
                    "input": [
                        {
                            "name": "NpuModel",
                            "file": "npu.py",
                            "optimize": "size",
                            "parameters": {"input-shape": "1x4"},
                        },
                        {"name": "CpuModel", "file": "cpu.pte", "cpu": None},
                    ]
                },
            }
            models = generator.model_inputs(mlops, root)

        self.assertEqual([model.name for model in models], ["NpuModel", "CpuModel"])
        self.assertEqual(models[0].parameters, {"input_shape": (1, 4)})
        self.assertFalse(models[0].cpu_only)
        self.assertTrue(models[1].cpu_only)
        self.assertTrue(models[1].imported)

    def test_model_inputs_reject_plural_property(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.py").write_text("# model", encoding="utf-8")
            mlops = {
                "processor": {"type": "Cortex-M55"},
                "model": {
                    "inputs": [{"name": "TinyCNN", "file": "model.py"}],
                },
            }

            with self.assertRaisesRegex(SystemExit, "singular model.input"):
                generator.model_inputs(mlops, root)

    def test_model_inputs_accept_pt2_and_reject_inapplicable_properties(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "models.pt2").write_bytes(b"PT2")
            (root / "program.pte").write_bytes(b"PTE")
            mlops = {
                "processor": {"type": "Cortex-M55"},
                "npu": {"type": "Ethos-U55"},
                "model": {"input": [{"name": "Models", "file": "models.pt2"}]},
            }
            inputs = generator.model_inputs(mlops, root)
            self.assertEqual(inputs[0].input_type, ".pt2")

            mlops["model"]["input"] = [
                {"name": "Program", "file": "program.pte", "optimize": "size"}
            ]
            with self.assertRaisesRegex(SystemExit, "optimize is invalid for .pte"):
                generator.model_inputs(mlops, root)

            mlops["model"]["input"] = [
                {"name": "Models", "file": "models.pt2", "parameters": {"shape": "1x4"}}
            ]
            with self.assertRaisesRegex(SystemExit, "parameters are valid only for .py"):
                generator.model_inputs(mlops, root)

    def test_input_case_requires_explicit_args_and_kwargs(self) -> None:
        args, kwargs = generator._input_case(((1, 2), {"scale": 3}), "example")
        self.assertEqual(args, (1, 2))
        self.assertEqual(kwargs, {"scale": 3})
        with self.assertRaisesRegex(SystemExit, r"must return \(args, kwargs\)"):
            generator._input_case((1, 2, 3), "example")

    def test_imported_pte_must_match_declared_execution(self) -> None:
        cpu = generator.ModelInput("CPU", "CPU", Path("cpu.pte"), {}, True, None, True)
        npu = generator.ModelInput("NPU", "NPU", Path("npu.pte"), {}, False, None, True)
        delegated = generator.PteInspection(("forward",), ("EthosUBackend",), ("ethos-u55-128",), (), ())
        portable = generator.PteInspection(("forward",), (), (), (), ())
        mismatched = generator.PteInspection(("forward",), ("EthosUBackend",), ("ethos-u85-256",), (), ())
        with mock.patch.object(generator, "inspect_pte", return_value=delegated), self.assertRaisesRegex(
            SystemExit, "CPU-only input contains"
        ):
            generator.validate_pte_target(cpu, b"PTE", "ethos-u55-128")
        with mock.patch.object(generator, "inspect_pte", return_value=portable), self.assertRaisesRegex(
            SystemExit, "contains no Ethos-U delegate"
        ):
            generator.validate_pte_target(npu, b"PTE", "ethos-u55-128")
        with mock.patch.object(generator, "inspect_pte", return_value=mismatched), self.assertRaisesRegex(
            SystemExit, "not ethos-u55-128"
        ):
            generator.validate_pte_target(npu, b"PTE", "ethos-u55-128")
        with mock.patch.object(generator, "inspect_pte", return_value=delegated), self.assertRaisesRegex(
            SystemExit, "system-config"
        ):
            generator.validate_pte_target(
                npu, b"PTE", "ethos-u55-128", {"system-config": "Other_System"}
            )
        unknown = generator.PteInspection(("forward",), ("XnnpackBackend",), (), (), ())
        with mock.patch.object(generator, "inspect_pte", return_value=unknown), self.assertRaisesRegex(
            SystemExit, "unsupported backend delegates"
        ):
            generator.validate_pte_target(npu, b"PTE", "ethos-u55-128")

    @unittest.skipUnless(importlib.util.find_spec("executorch"), "requires project ExecuTorch environment")
    def test_real_pte_rejects_malformed_and_mismatched_programs(self) -> None:
        import torch
        from executorch.exir import to_edge

        class PortableModel(torch.nn.Module):
            def forward(self, value):
                return value + 1

        portable_export = torch.export.export(PortableModel(), (torch.ones(1),))
        portable_pte = bytes(to_edge(portable_export).to_executorch().buffer)
        npu = generator.ModelInput("NPU", "NPU", Path("npu.pte"), {}, False, None, True)
        cpu = generator.ModelInput("CPU", "CPU", Path("cpu.pte"), {}, True, None, True)

        with self.assertRaisesRegex(SystemExit, "invalid ExecuTorch program"):
            generator.validate_pte_target(cpu, b"not a PTE", None)
        with self.assertRaisesRegex(SystemExit, "contains no Ethos-U delegate"):
            generator.validate_pte_target(npu, portable_pte, "ethos-u55-128")

        mlops = {
            "npu": {"type": "Ethos-U55", "macs": 128},
            "model": {},
            "vela": {
                "options": (
                    "--system-config Ethos_U55_High_End_Embedded "
                    "--memory-mode Shared_Sram"
                )
            },
        }
        with redirect_stdout(io.StringIO()):
            spec = generator.compile_spec(mlops, ROOT)
            ethos_pte = generator.export_model(
                spec,
                ROOT / "model" / "TinyCNN" / "model.py",
                {"calibration_samples": 1},
                "TinyCNN",
            )

        inspection = generator.validate_pte_target(
            npu,
            ethos_pte,
            "ethos-u55-128",
            {
                "system-config": "Ethos_U55_High_End_Embedded",
                "memory-mode": "Shared_Sram",
            },
        )
        self.assertEqual(inspection.methods, ("forward",))

        with self.assertRaisesRegex(SystemExit, "CPU-only input contains"):
            generator.validate_pte_target(cpu, ethos_pte, None)
        with self.assertRaisesRegex(SystemExit, "not ethos-u85-256"):
            generator.validate_pte_target(npu, ethos_pte, "ethos-u85-256")
        with self.assertRaisesRegex(SystemExit, "system-config"):
            generator.validate_pte_target(
                npu, ethos_pte, "ethos-u55-128", {"system-config": "Other_System"}
            )
        with self.assertRaisesRegex(SystemExit, "memory-mode"):
            generator.validate_pte_target(
                npu, ethos_pte, "ethos-u55-128", {"memory-mode": "Other_Memory"}
            )

    def test_executorch_version_prefers_exact_selected_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mlops = root / "demo.cbuild-mlops.yml"
            pack = root / "demo.cbuild-pack.yml"
            pack.write_text(
                yaml.safe_dump(
                    {
                        "cbuild-pack": {
                            "resolved-packs": [
                                {
                                    "resolved-pack": "PyTorch::ExecuTorch@1.4.0",
                                    "selected-by-pack": ["PyTorch::ExecuTorch"],
                                },
                                {
                                    "resolved-pack": "PyTorch::ExecuTorch@1.5.1",
                                    "selected-by-pack": ["PyTorch::ExecuTorch@1.5.1"],
                                },
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(generator.executorch_version(mlops), "1.5.1")

    def test_pack_prerelease_suffix_uses_base_python_version(self) -> None:
        with mock.patch.object(
            generator.importlib.metadata, "version", return_value="1.6.0"
        ):
            generator.check_executorch_version("1.6.0-zephyrpreview1")

    def test_components_select_known_operators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pack = Path(temporary)
            (pack / "ExecuTorch.pdsc").write_text(
                '<package><components>'
                '<component Cgroup="ExecuTorch Operators" Csub="Portable add"/>'
                '<component Cgroup="ExecuTorch Operators" Csub="Quantized dequantize"/>'
                '</components></package>',
                encoding="utf-8",
            )
            runtime, operators, cmsis_nn = generator.components(
                b"aten::add quantized_decomposed::dequantize_per_tensor",
                pack,
                delegated=True,
            )
        self.assertEqual(
            runtime, ["Runtime", "Kernel Utils", "Kernel Registration", "Backend EthosU"]
        )
        self.assertEqual(operators, ["Portable add", "Quantized dequantize"])
        self.assertFalse(cmsis_nn)

    def test_cpu_components_select_cmsis_nn_without_ethosu_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pack = Path(temporary)
            (pack / "ExecuTorch.pdsc").write_text(
                '<package><conditions>'
                '<condition id="CMSIS-NN"/>'
                '<condition id="cortex-conv"><require condition="CMSIS-NN"/></condition>'
                '</conditions><components>'
                '<component Cgroup="ExecuTorch Operators" '
                'Csub="Cortex-M quantized_conv2d" condition="cortex-conv"/>'
                '<component Cgroup="ExecuTorch Operators" Csub="Portable view_copy"/>'
                '</components></package>',
                encoding="utf-8",
            )
            runtime, operators, cmsis_nn = generator.components(
                b"cortex_m::quantized_conv2d_nhwc dim_order_ops::view_copy",
                pack,
                delegated=False,
            )

        self.assertEqual(runtime, ["Runtime", "Kernel Utils", "Kernel Registration"])
        self.assertEqual(operators, ["Cortex-M quantized_conv2d", "Portable view_copy"])
        self.assertTrue(cmsis_nn)

    def test_generated_c_and_layer_content(self) -> None:
        source = generator.c_array(bytes([0, 1, 255]))
        self.assertIn("__attribute__((aligned(16)))", source)
        self.assertIn("0x00, 0x01, 0xff", source)

        layer = generator.clayer(
            {"description": "Demo", "model": {"name": "Network"}},
            ["Runtime"],
            ["Portable add"],
            False,
            Path("demo.cbuild-mlops.yml"),
            "1.5.1",
        )
        self.assertIn("pack: PyTorch::ExecuTorch@1.5.1", layer)
        self.assertIn("component: Machine Learning:ExecuTorch Operators:Portable add", layer)
        self.assertIn("file: ./model_pte.c", layer)

    def test_cpu_layer_adds_unversioned_cmsis_nn(self) -> None:
        layer = generator.clayer(
            {"description": "CPU", "model": {"name": "Network"}},
            ["Runtime"],
            ["Cortex-M quantized_conv2d"],
            True,
            Path("demo.cbuild-mlops.yml"),
            "1.5.1",
        )
        self.assertIn("pack: ARM::CMSIS-NN\n", layer)
        self.assertNotIn("pack: ARM::CMSIS-NN@", layer)
        self.assertIn("component: CMSIS:NN Lib", layer)
        self.assertNotIn("Backend EthosU", layer)

    def test_vela_summary_records_effective_configuration_and_report(self) -> None:
        summary = generator.vela_summary(
            "TinyCNN", self.compile_spec(), "Network summary\r\nTotal SRAM used 6 KiB\r\n"
        )
        self.assertIn("# Vela report: TinyCNN", summary)
        self.assertIn("accelerator: `ethos-u55-256`", summary)
        self.assertIn("system config: `Test_System`", summary)
        self.assertIn("memory mode: `Shared_Sram`", summary)
        self.assertIn("```text\nNetwork summary\nTotal SRAM used 6 KiB\n```", summary)
        self.assertNotIn("\r", summary)

    def test_replace_outputs_preserves_destinations_when_staging_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layer = Path(temporary) / "ai_layer"
            layer.mkdir()
            existing = layer / "model.pte"
            existing.write_bytes(b"old")
            with self.assertRaisesRegex(TypeError, "unsupported generated output type"):
                generator.replace_outputs(layer, {"model.pte": b"new", "bad": object()})
            self.assertEqual(existing.read_bytes(), b"old")
            self.assertFalse((layer / "bad").exists())

    def test_replace_outputs_leaves_files_not_named_by_current_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layer = Path(temporary) / "ai_layer"
            layer.mkdir()
            stale = layer / "RemovedModel.pte"
            stale.write_bytes(b"stale")

            generator.replace_outputs(layer, {"CurrentModel.pte": b"current"})

            self.assertEqual((layer / "CurrentModel.pte").read_bytes(), b"current")
            self.assertEqual(stale.read_bytes(), b"stale")
            self.assertFalse((layer / ".ai_layer_manifest.json").exists())

    def test_zephyr_metadata_owns_model_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pack = Path(temporary)
            (pack / "zephyr").mkdir()
            (pack / "zephyr" / "Kconfig").write_text(
                "config CMSIS_EXECUTORCH_OP_QUANTIZED_DEQUANTIZE\n", encoding="utf-8"
            )
            metadata = generator.zephyr_metadata(
                ["Quantized dequantize"], pack, delegated=True
            )

        self.assertIn("target_sources(app PRIVATE", metadata["CMakeLists.txt"])
        self.assertIn("target_include_directories(app PRIVATE", metadata["CMakeLists.txt"])
        self.assertNotIn("INTERFACE_SOURCES", metadata["CMakeLists.txt"])
        self.assertNotIn("cmake_language(DEFER", metadata["CMakeLists.txt"])
        self.assertNotIn("EXTRA_CONF_FILE", metadata["CMakeLists.txt"])
        self.assertEqual(metadata["CMakeLists.txt"].count("model_pte.c"), 1)
        self.assertIn(
            "CONFIG_CMSIS_EXECUTORCH_OP_QUANTIZED_DEQUANTIZE=y",
            metadata["operators.conf"],
        )
        self.assertIn("CONFIG_CMSIS_EXECUTORCH_BACKEND_ETHOS_U=y", metadata["components.conf"])

    def test_cpu_zephyr_metadata_omits_ethosu_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pack = Path(temporary)
            (pack / "zephyr").mkdir()
            (pack / "zephyr" / "Kconfig").write_text(
                "config CMSIS_EXECUTORCH_OP_CORTEX_M_QUANTIZED_CONV2D\n",
                encoding="utf-8",
            )
            metadata = generator.zephyr_metadata(
                ["Cortex-M quantized_conv2d"], pack, delegated=False
            )

        self.assertNotIn("BACKEND_ETHOS_U", metadata["components.conf"])
        self.assertIn(
            "CONFIG_CMSIS_EXECUTORCH_OP_CORTEX_M_QUANTIZED_CONV2D=y",
            metadata["operators.conf"],
        )

    def test_zephyr_templates_configure_layer_before_adding_sources(self) -> None:
        for template in ("zephyr-m", "zephyr-u55", "zephyr-u85"):
            with self.subTest(template=template):
                cmake = (ROOT / template / "CMakeLists.txt").read_text(encoding="utf-8")
                configure = cmake.index("list(APPEND EXTRA_CONF_FILE")
                find_zephyr = cmake.index("find_package(Zephyr")
                add_layer = cmake.index("add_subdirectory(ai_layer)")
                self.assertLess(configure, find_zephyr)
                self.assertLess(find_zephyr, add_layer)

    def test_clayer_uses_absolute_mlops_path_across_windows_drives(self) -> None:
        mlops_file = Path("C:/temporary/demo.cbuild-mlops.yml")
        with mock.patch.object(generator.os.path, "relpath", side_effect=ValueError):
            layer = generator.clayer(
                {"description": "Demo", "model": {"name": "Network"}},
                ["Runtime"],
                ["Portable add"],
                False,
                mlops_file,
                "1.5.1",
            )

        self.assertIn(f"create_ai_layer.py {mlops_file}`", layer)

    def test_main_writes_all_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            solution_dir = Path(temporary) / "solutions" / "demo"
            solution_dir.mkdir(parents=True)
            mlops_file = solution_dir / "demo.cbuild-mlops.yml"
            mlops_file.write_text(
                yaml.safe_dump(
                    {
                        "cbuild-mlops": {
                            "description": "Demo",
                            "npu": {"type": "Ethos-U55"},
                            "model": {
                                "clayer": "generated/ai_layer.clayer.yml",
                                "name": "Network",
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            argv = [str(SCRIPT), str(mlops_file)]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                generator, "run_in_venv"
            ) as run_in_venv, mock.patch.object(
                generator, "compile_spec", return_value=self.compile_spec()
            ), mock.patch.object(
                generator, "export_model", return_value=b"PTE"
            ) as export_model, mock.patch.object(
                generator, "executorch_version", return_value="1.5.1"
            ), mock.patch.object(
                generator, "executorch_pack", return_value=solution_dir / "pack"
            ), mock.patch.object(
                generator, "check_executorch_version"
            ), mock.patch.object(
                generator, "generate_operator_lists",
                return_value=({"Network": ["aten::add.out"]}, ["aten::add.out"]),
            ), mock.patch.object(
                generator, "operator_components", return_value=(["Portable add"], False)
            ), mock.patch.object(
                generator, "validate_pte_target", return_value=self.inspection("forward", delegated=True)
            ):
                generator.main()

            run_in_venv.assert_called_once_with(ROOT)
            export_model.assert_called_once_with(
                mock.ANY, (ROOT / "model" / "TinyCNN" / "model.py").resolve(), {}, "Network"
            )

            output = solution_dir / "generated"
            self.assertEqual((output / "model.pte").read_bytes(), b"PTE")
            self.assertTrue((output / "model_pte.c").is_file())
            self.assertTrue((output / "model_pte.h").is_file())
            self.assertTrue((output / "ai_layer.clayer.yml").is_file())
            self.assertTrue((output / "model_REPORT.md").is_file())

    def test_main_accepts_empty_zephyr_and_writes_module_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            solution_dir = Path(temporary) / "solutions" / "demo"
            solution_dir.mkdir(parents=True)
            mlops_file = solution_dir / "demo.cbuild-mlops.yml"
            mlops_file.write_text(
                yaml.safe_dump(
                    {
                        "cbuild-mlops": {
                            "description": "Demo",
                            "npu": {"type": "Ethos-U55"},
                            "model": {
                                "clayer": "generated/ai_layer.clayer.yml",
                                # Equivalent to a plain ``zephyr:`` YAML node.
                                "zephyr": None,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            argv = [str(SCRIPT), str(mlops_file)]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                generator, "run_in_venv"
            ), mock.patch.object(generator, "compile_spec", return_value=self.compile_spec()), mock.patch.object(
                generator, "export_model", return_value=b"PTE"
            ), mock.patch.object(generator, "executorch_version", return_value="1.6.0"), mock.patch.object(
                generator, "executorch_pack", return_value=solution_dir / "pack"
            ), mock.patch.object(
                generator, "check_executorch_version"
            ), mock.patch.object(
                generator, "generate_operator_lists",
                return_value=({"Model": ["quantized_decomposed::dequantize_per_tensor"]}, ["quantized_decomposed::dequantize_per_tensor"]),
            ), mock.patch.object(
                generator, "operator_components", return_value=(["Quantized dequantize"], False)
            ), mock.patch.object(
                generator,
                "zephyr_metadata",
                return_value={
                    "CMakeLists.txt": generator.ZEPHYR_CMAKE,
                    "components.conf": generator.ZEPHYR_COMPONENTS,
                    "operators.conf": "CONFIG_CMSIS_EXECUTORCH_OP_QUANTIZED_DEQUANTIZE=y\n",
                },
            ), mock.patch.object(
                generator, "validate_pte_target", return_value=self.inspection("forward", delegated=True)
            ):
                generator.main()

            output = solution_dir / "generated"
            self.assertTrue((output / "CMakeLists.txt").is_file())
            self.assertTrue((output / "components.conf").is_file())
            self.assertTrue((output / "operators.conf").is_file())
            self.assertTrue((output / "ai_layer.clayer.yml").is_file())

    def test_main_generates_cpu_layer_without_vela_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            solution_dir = Path(temporary) / "cmsis-m"
            solution_dir.mkdir()
            mlops_file = solution_dir / "cmsis-m.cbuild-mlops.yml"
            mlops_file.write_text(
                yaml.safe_dump(
                    {
                        "cbuild-mlops": {
                            "description": "CPU",
                            "processor": {"type": "Cortex-M55"},
                            "npu": {"type": "Ethos-U55", "macs": 256},
                            "model": {
                                "clayer": "ai_layer/ai_layer.clayer.yml",
                                "cpu": "Cortex-M55",
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            target = mock.Mock()
            argv = [str(SCRIPT), str(mlops_file)]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                generator, "run_in_venv"
            ), mock.patch.object(generator, "cpu_target", return_value=target), mock.patch.object(
                generator, "export_model_cpu", return_value=b"CPU"
            ) as export_cpu, mock.patch.object(
                generator, "executorch_version", return_value="1.5.1"
            ), mock.patch.object(
                generator, "executorch_pack", return_value=solution_dir / "pack"
            ), mock.patch.object(
                generator, "check_executorch_version"
            ), mock.patch.object(
                generator, "generate_operator_lists",
                return_value=({"Model": ["cortex_m::quantized_conv2d_nhwc"]}, ["cortex_m::quantized_conv2d_nhwc"]),
            ), mock.patch.object(
                generator, "operator_components", return_value=(["Cortex-M quantized_conv2d"], True)
            ), mock.patch.object(
                generator, "validate_pte_target", return_value=self.inspection("forward")
            ):
                generator.main()

            export_cpu.assert_called_once_with(
                target, (ROOT / "model" / "TinyCNN" / "model.py").resolve(), {}, "Model"
            )
            output = solution_dir / "ai_layer"
            self.assertEqual((output / "model.pte").read_bytes(), b"CPU")
            self.assertFalse((output / "VELA_SUMMARY.md").exists())
            self.assertTrue((output / "model_REPORT.md").exists())
            layer = (output / "ai_layer.clayer.yml").read_text(encoding="utf-8")
            self.assertIn("pack: ARM::CMSIS-NN", layer)

    def test_main_rejects_non_mapping_zephyr_setting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            solution_dir = Path(temporary)
            mlops_file = solution_dir / "demo.cbuild-mlops.yml"
            mlops_file.write_text(
                yaml.safe_dump(
                    {
                        "cbuild-mlops": {
                            "model": {
                                "clayer": "ai_layer/ai_layer.clayer.yml",
                                "zephyr": True,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            argv = [str(SCRIPT), str(mlops_file)]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                generator, "run_in_venv"
            ), self.assertRaisesRegex(SystemExit, "model.zephyr must be empty or a mapping"):
                generator.main()

    def test_main_generates_multiple_models_and_combined_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            solution = Path(temporary)
            for name in ("npu.py", "cpu.py"):
                (solution / name).write_text("# mocked", encoding="utf-8")
            mlops_file = solution / "demo.cbuild-mlops.yml"
            mlops_file.write_text(
                yaml.safe_dump({
                    "cbuild-mlops": {
                        "processor": {"type": "Cortex-M55"},
                        "npu": {"type": "Ethos-U55"},
                        "model": {
                            "clayer": "ai_layer/ai_layer.clayer.yml",
                            "zephyr": {},
                            "input": [
                                {"name": "NpuModel", "file": "npu.py"},
                                {"name": "CpuModel", "file": "cpu.py", "cpu": None},
                            ],
                        },
                    }
                }),
                encoding="utf-8",
            )
            npu_spec = self.compile_spec()

            def compile_for_model(_mlops, _directory, *, cpu_only=False, optimize=None):
                return None if cpu_only else npu_spec

            def map_components(names, _pack):
                if any(name.startswith("cortex_m::") for name in names):
                    return ["Cortex-M quantized_conv2d"], True
                if len(names) > 1:
                    return ["Cortex-M quantized_conv2d", "Portable add"], True
                return ["Portable add"], False

            argv = [str(SCRIPT), str(mlops_file)]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                generator, "run_in_venv"
            ), mock.patch.object(generator, "executorch_version", return_value="1.5.1"), mock.patch.object(
                generator, "check_executorch_version"
            ), mock.patch.object(generator, "executorch_pack", return_value=solution / "pack"), mock.patch.object(
                generator, "compile_spec", side_effect=compile_for_model
            ), mock.patch.object(generator, "cpu_target", return_value=mock.Mock()), mock.patch.object(
                generator, "export_model", return_value=b"NPU"
            ), mock.patch.object(generator, "export_model_cpu", return_value=b"CPU"), mock.patch.object(
                generator,
                "generate_operator_lists",
                return_value=(
                    {
                        "NpuModel": ["aten::add.out"],
                        "CpuModel": ["cortex_m::quantized_conv2d_nhwc"],
                    },
                    ["aten::add.out", "cortex_m::quantized_conv2d_nhwc"],
                ),
            ), mock.patch.object(generator, "operator_components", side_effect=map_components), mock.patch.object(
                generator,
                "zephyr_metadata",
                side_effect=lambda operators, pack, delegated, models: {
                    "CMakeLists.txt": generator.zephyr_cmake(models),
                    "components.conf": "components\n",
                    "operators.conf": "operators\n",
                },
            ), mock.patch.object(
                generator,
                "validate_pte_target",
                side_effect=[self.inspection("forward", delegated=True), self.inspection("forward")],
            ):
                generator.main()

            output = solution / "ai_layer"
            self.assertEqual((output / "NpuModel.pte").read_bytes(), b"NPU")
            self.assertEqual((output / "CpuModel.pte").read_bytes(), b"CPU")
            self.assertTrue((output / "NpuModel_REPORT.md").is_file())
            self.assertTrue((output / "CpuModel_REPORT.md").is_file())
            layer = (output / "ai_layer.clayer.yml").read_text(encoding="utf-8")
            self.assertIn("group: NpuModel", layer)
            self.assertIn("group: CpuModel", layer)
            self.assertEqual(layer.count("Backend EthosU"), 1)
            cmake = (output / "CMakeLists.txt").read_text(encoding="utf-8")
            self.assertEqual(cmake.count("NpuModel_pte.c"), 1)
            self.assertEqual(cmake.count("CpuModel_pte.c"), 1)

    def test_main_lowers_multi_program_pt2_to_one_pte(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            solution = Path(temporary)
            (solution / "models.pt2").write_bytes(b"PT2")
            mlops_file = solution / "demo.cbuild-mlops.yml"
            mlops_file.write_text(
                yaml.safe_dump({
                    "cbuild-mlops": {
                        "processor": {"type": "Cortex-M55"},
                        "model": {
                            "clayer": "ai_layer/ai_layer.clayer.yml",
                            "input": [{"name": "Models", "file": "models.pt2"}],
                        },
                    }
                }),
                encoding="utf-8",
            )
            inspection = self.inspection("detector", "classifier")
            argv = [str(SCRIPT), str(mlops_file)]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                generator, "run_in_venv"
            ), mock.patch.object(generator, "executorch_version", return_value="1.5.1"), mock.patch.object(
                generator, "check_executorch_version"
            ), mock.patch.object(generator, "executorch_pack", return_value=solution / "pack"), mock.patch.object(
                generator, "compile_spec", return_value=None
            ), mock.patch.object(generator, "cpu_target", return_value=mock.sentinel.target), mock.patch.object(
                generator, "lower_pt2", return_value=(b"MULTI-PTE", ("detector", "classifier"))
            ) as lower, mock.patch.object(
                generator, "validate_pte_target", return_value=inspection
            ), mock.patch.object(
                generator, "generate_operator_lists", return_value=({"Models": []}, [])
            ), mock.patch.object(generator, "operator_components", return_value=([], False)):
                generator.main()

            lower.assert_called_once_with(
                None, mock.sentinel.target, (solution / "models.pt2").resolve(), "Models"
            )
            output = solution / "ai_layer"
            self.assertEqual((output / "Models.pte").read_bytes(), b"MULTI-PTE")
            report = (output / "Models_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("detector, classifier", report)

    def test_prebuilt_pte_is_not_lowered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "program.pte"
            source.write_bytes(b"PTE")
            mlops_file = root / "demo.cbuild-mlops.yml"
            mlops_file.write_text(
                yaml.safe_dump({
                    "cbuild-mlops": {
                        "npu": {"type": "Ethos-U55"},
                        "model": {
                            "clayer": "ai_layer/ai_layer.clayer.yml",
                            "input": [{"name": "Program", "file": "program.pte"}],
                        },
                    }
                }), encoding="utf-8"
            )
            argv = [str(SCRIPT), str(mlops_file)]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                generator, "run_in_venv"
            ), mock.patch.object(generator, "executorch_version", return_value="1.5.1"), mock.patch.object(
                generator, "check_executorch_version"
            ), mock.patch.object(generator, "executorch_pack", return_value=root / "pack"), mock.patch.object(
                generator, "compile_spec"
            ) as compile_spec, mock.patch.object(
                generator, "lower_pt2"
            ) as lower, mock.patch.object(
                generator, "validate_pte_target", return_value=self.inspection("forward", delegated=True)
            ), mock.patch.object(
                generator, "generate_operator_lists", return_value=({"Program": []}, [])
            ), mock.patch.object(generator, "operator_components", return_value=([], False)):
                generator.main()

            compile_spec.assert_not_called()
            lower.assert_not_called()
            self.assertEqual((root / "ai_layer" / "Program.pte").read_bytes(), b"PTE")

    def test_input_option_uses_unchanged_pte_for_operator_union(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            solution = Path(temporary)
            (solution / "a-source.pte").write_bytes(b"SOURCE-A")
            (solution / "b-source.pte").write_bytes(b"NEW-B")
            layer_dir = solution / "ai_layer"
            layer_dir.mkdir()
            (layer_dir / "A.pte").write_bytes(b"EXISTING-A")
            (layer_dir / "A_REPORT.md").write_text("keep A\n", encoding="utf-8")
            mlops_file = solution / "demo.cbuild-mlops.yml"
            mlops_file.write_text(
                yaml.safe_dump({
                    "cbuild-mlops": {
                        "npu": {"type": "Ethos-U55"},
                        "model": {
                            "clayer": "ai_layer/ai_layer.clayer.yml",
                            "input": [
                                {"name": "A", "file": "a-source.pte"},
                                {"name": "B", "file": "b-source.pte"},
                            ],
                        },
                    }
                }),
                encoding="utf-8",
            )

            def operator_lists(paths, _stage):
                # Windows runners can spell the same temporary directory with
                # either its long name or its DOS 8.3 alias. Compare filesystem
                # identity instead of the textual path representation.
                self.assertTrue(paths["A"].samefile(layer_dir / "A.pte"))
                self.assertEqual(paths["A"].read_bytes(), b"EXISTING-A")
                self.assertEqual(paths["B"].read_bytes(), b"NEW-B")
                return ({"A": ["aten::add.out"], "B": ["aten::mul.out"]},
                        ["aten::add.out", "aten::mul.out"])

            argv = [str(SCRIPT), str(mlops_file), "--input", "B"]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                generator, "run_in_venv"
            ), mock.patch.object(generator, "executorch_version", return_value="1.5.1"), mock.patch.object(
                generator, "check_executorch_version"
            ), mock.patch.object(generator, "executorch_pack", return_value=solution / "pack"), mock.patch.object(
                generator, "compile_spec", return_value=self.compile_spec()
            ), mock.patch.object(generator, "generate_operator_lists", side_effect=operator_lists), mock.patch.object(
                generator, "operator_components", return_value=(["Portable add"], False)
            ), mock.patch.object(
                generator, "validate_pte_target", return_value=self.inspection("forward", delegated=True)
            ):
                generator.main()

            self.assertEqual((layer_dir / "A.pte").read_bytes(), b"EXISTING-A")
            self.assertEqual((layer_dir / "A_REPORT.md").read_text(encoding="utf-8"), "keep A\n")
            self.assertEqual((layer_dir / "B.pte").read_bytes(), b"NEW-B")
            self.assertTrue((layer_dir / "B_REPORT.md").is_file())


if __name__ == "__main__":
    unittest.main()
