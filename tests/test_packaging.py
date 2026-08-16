"""Smoke tests for packaging.

These are deliberately boring: they fail loudly when the src-layout install is
broken (wrong package path, missing py.typed, dead console script), which is the
single most common way a Python repo is unusable for a stranger who clones it.
"""

from importlib.metadata import version as pkg_version
from importlib.resources import files

from typer.testing import CliRunner

import citely
from citely.cli import app

runner = CliRunner()


def test_version_matches_installed_distribution() -> None:
    assert citely.__version__ == pkg_version("citely")
    assert citely.__version__ != "0.0.0.dev0", "package is not installed; run `make install`"


def test_package_ships_py_typed_marker() -> None:
    assert files("citely").joinpath("py.typed").is_file()


def test_cli_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == citely.__version__


def test_cli_shows_help_without_args() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code != 0
    assert "Usage" in result.stdout
