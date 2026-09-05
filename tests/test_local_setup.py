from pathlib import Path

import pytest

from src.local_setup import _safe_target, runtime_asset_name


def test_runtime_asset_name_for_supported_platforms():
    assert runtime_asset_name("Windows", "AMD64") == "llama-{tag}-bin-win-cpu-x64.zip"
    assert runtime_asset_name("Windows", "ARM64") == "llama-{tag}-bin-win-cpu-arm64.zip"
    assert runtime_asset_name("Linux", "x86_64") == "llama-{tag}-bin-ubuntu-x64.tar.gz"
    assert runtime_asset_name("Darwin", "arm64") == "llama-{tag}-bin-macos-arm64.tar.gz"


def test_runtime_asset_rejects_unknown_platform():
    with pytest.raises(RuntimeError, match="unsupported operating system"):
        runtime_asset_name("Plan9", "x86_64")


def test_archive_path_cannot_escape_destination(tmp_path: Path):
    with pytest.raises(RuntimeError, match="unsafe archive path"):
        _safe_target(tmp_path, "../../outside")
