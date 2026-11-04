#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Tests for model_conversion/create_ai_layer.py."""

from __future__ import annotations

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

    def test_components_select_known_operators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pack = Path(temporary)
            (pack / "ExecuTorch.pdsc").write_text(
                '<component Csub="Portable add"/><component Csub="Quantized dequantize"/>',
                encoding="utf-8",
            )
            runtime, operators = generator.components(
                b"aten::add quantized_decomposed::dequantize_per_tensor", pack
            )
        self.assertEqual(
            runtime, ["Runtime", "Kernel Utils", "Kernel Registration", "Backend EthosU"]
        )
        self.assertEqual(operators, ["Portable add", "Quantized dequantize"])

    def test_generated_c_and_layer_content(self) -> None:
        source = generator.c_array(bytes([0, 1, 255]))
        self.assertIn("__attribute__((aligned(16)))", source)
        self.assertIn("0x00, 0x01, 0xff", source)

        layer = generator.clayer(
            {"description": "Demo", "model": {"name": "Network"}},
            ["Runtime"],
            ["Portable add"],
            Path("demo.cbuild-mlops.yml"),
            "1.5.1",
        )
        self.assertIn("pack: PyTorch::ExecuTorch@1.5.1", layer)
        self.assertIn("component: Machine Learning:ExecuTorch Operators:Portable add", layer)
        self.assertIn("file: ./model_pte.c", layer)

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

    def test_zephyr_metadata_owns_model_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pack = Path(temporary)
            (pack / "zephyr").mkdir()
            (pack / "zephyr" / "Kconfig").write_text(
                "config CMSIS_EXECUTORCH_OP_QUANTIZED_DEQUANTIZE\n", encoding="utf-8"
            )
            metadata = generator.zephyr_metadata(["Quantized dequantize"], pack)

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

    def test_zephyr_templates_configure_layer_before_adding_sources(self) -> None:
        for template in ("zephyr-u55", "zephyr-u85"):
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
                generator, "components", return_value=(["Runtime"], ["Portable add"])
            ):
                generator.main()

            run_in_venv.assert_called_once_with(ROOT)
            export_model.assert_called_once_with(mock.ANY, {}, solution_dir.resolve())

            output = solution_dir / "generated"
            self.assertEqual((output / "model.pte").read_bytes(), b"PTE")
            self.assertTrue((output / "model_pte.c").is_file())
            self.assertTrue((output / "model_pte.h").is_file())
            self.assertTrue((output / "ai_layer.clayer.yml").is_file())
            self.assertTrue((output / "VELA_SUMMARY.md").is_file())

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
                generator, "components", return_value=(["Runtime"], ["Quantized dequantize"])
            ), mock.patch.object(
                generator,
                "zephyr_metadata",
                return_value={
                    "CMakeLists.txt": generator.ZEPHYR_CMAKE,
                    "components.conf": generator.ZEPHYR_COMPONENTS,
                    "operators.conf": "CONFIG_CMSIS_EXECUTORCH_OP_QUANTIZED_DEQUANTIZE=y\n",
                },
            ):
                generator.main()

            output = solution_dir / "generated"
            self.assertTrue((output / "CMakeLists.txt").is_file())
            self.assertTrue((output / "components.conf").is_file())
            self.assertTrue((output / "operators.conf").is_file())
            self.assertTrue((output / "ai_layer.clayer.yml").is_file())

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


if __name__ == "__main__":
    unittest.main()
