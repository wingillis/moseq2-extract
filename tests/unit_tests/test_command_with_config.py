"""
Unit tests for the restored release-era config-file behavior:

- ``--config-file`` is a subcommand-level option on the commands that support
  it (find-roi, extract, batch-extract) and reads a FLAT yaml file (option
  names at the top level, no ``extract:`` section).
- Parameter precedence: CLI options > config file values > click defaults.
- ``generate-config`` writes a flat yaml of the extract command defaults.
"""

import os
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from click.testing import CliRunner

import moseq2_extract.cli as cli_module
from moseq2_extract.cli import cli, extract, generate_config
from moseq2_extract.util import read_yaml


class TestCommandWithConfig(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        # fake depth file so the exists=True argument check passes
        self.input_file = Path(self.tmp.name) / "depth.dat"
        self.input_file.touch()

        self.captured = {}

        def fake_extract_wrapper(input_file, output_dir, config_data, **kwargs):
            self.captured["config_data"] = dict(config_data)
            self.captured["output_dir"] = output_dir

        self._orig_wrapper = cli_module.extract_wrapper
        cli_module.extract_wrapper = fake_extract_wrapper
        self.addCleanup(
            setattr, cli_module, "extract_wrapper", self._orig_wrapper
        )

    def invoke_extract(self, extra_args):
        return self.runner.invoke(
            extract, [str(self.input_file)] + extra_args, catch_exceptions=False
        )

    def test_extract_has_config_file_option(self):
        result = self.runner.invoke(extract, ["--help"])
        self.assertIn("--config-file", result.output)

    def test_group_has_no_config_file_option(self):
        result = self.runner.invoke(cli, ["--help"])
        self.assertNotIn("--config-file", result.output.split("Commands")[0])

    def test_defaults_without_config(self):
        result = self.invoke_extract([])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(self.captured["config_data"]["min_height"], 10)

    def test_flat_config_overrides_default(self):
        config_file = Path(self.tmp.name) / "config.yaml"
        config_file.write_text(
            "min_height: 15\nbg_roi_dilate: [5, 5]\n"
            "output_dir: /should/be/ignored\nflip_classifier: null\n"
            "unknown_key: 1\n"
        )

        result = self.invoke_extract(["--config-file", str(config_file)])
        self.assertEqual(result.exit_code, 0)

        cfg = self.captured["config_data"]
        # config value wins over the default
        self.assertEqual(cfg["min_height"], 15)
        # yaml sequences are converted to tuples for tuple-typed options
        self.assertEqual(tuple(cfg["bg_roi_dilate"]), (5, 5))
        # output_dir from the config is not applied (avoids clobbering paths)
        self.assertEqual(self.captured["output_dir"], "proc")
        # null values are not applied (default None kept)
        self.assertIsNone(cfg["flip_classifier"])

    def test_cli_overrides_config(self):
        config_file = Path(self.tmp.name) / "config.yaml"
        config_file.write_text("min_height: 15\n")

        result = self.invoke_extract(
            ["--config-file", str(config_file), "--min-height", "20"]
        )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(self.captured["config_data"]["min_height"], 20)

    def test_sectioned_config_is_not_supported(self):
        # the new-style sectioned config has no effect on flat options
        config_file = Path(self.tmp.name) / "sectioned.yaml"
        config_file.write_text("extract:\n  min_height: 99\n")

        result = self.invoke_extract(["--config-file", str(config_file)])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(self.captured["config_data"]["min_height"], 10)


class TestGenerateConfig(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_generate_config_writes_flat_yaml(self):
        out = Path(self.tmp.name) / "config.yaml"
        result = self.runner.invoke(
            generate_config, ["--output-file", str(out)], catch_exceptions=False
        )
        self.assertEqual(result.exit_code, 0)

        data = read_yaml(out)
        # flat structure: option names at the top level, no extract section
        self.assertNotIn("extract", data)
        self.assertIn("min_height", data)
        self.assertEqual(data["min_height"], 10)
        self.assertIn("crop_size", data)
        self.assertEqual(list(data["crop_size"]), [80, 80])

        # flat config round-trips through the extract command
        input_file = Path(self.tmp.name) / "depth.dat"
        input_file.touch()
        captured = {}

        orig_wrapper = cli_module.extract_wrapper

        def fake_wrapper(input_file, output_dir, config_data, **kwargs):
            captured.update(config_data)

        cli_module.extract_wrapper = fake_wrapper
        try:
            result = self.runner.invoke(
                extract,
                [
                    str(input_file),
                    "--config-file",
                    str(out),
                    "--crop-size",
                    "40",
                    "40",
                ],
                catch_exceptions=False,
            )
            self.assertEqual(result.exit_code, 0)
            # config (k2 defaults) vs CLI override
            self.assertEqual(captured["min_height"], 10)
            self.assertEqual(tuple(captured["crop_size"]), (40, 40))
        finally:
            cli_module.extract_wrapper = orig_wrapper

    def test_generate_config_azure(self):
        out = Path(self.tmp.name) / "config.yaml"
        result = self.runner.invoke(
            generate_config,
            ["--output-file", str(out), "--camera-type", "azure"],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)

        data = read_yaml(out)
        self.assertEqual(list(data["bg_roi_depth_range"]), [550, 650])
        self.assertEqual(list(data["spatial_filter_size"]), [5])
        self.assertEqual(list(data["tail_filter_size"]), [15, 15])
        self.assertEqual(list(data["crop_size"]), [120, 120])
        self.assertEqual(data["camera_type"], "azure")


if __name__ == "__main__":
    unittest.main()
