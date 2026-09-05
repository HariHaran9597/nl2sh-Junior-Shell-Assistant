"""Download and install the local CPU runtime used by the nl2sh CLI.

The setup path intentionally uses only Python's standard library.  Hosted
inference is optional; users receive the model and run llama.cpp locally.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import tarfile
import urllib.request
import zipfile
from pathlib import Path


MODEL_REPO = "justhariharan/nl2sh-1.5b-Q4_K_M-GGUF"
MODEL_FILE = "nl2sh-1.5b.Q4_K_M.gguf"
# Pin the model commit and the raw downloadable object's SHA-256.  A changed
# artifact must be reviewed before setup accepts it as the model shipped by
# this release.
MODEL_REVISION = "6446a31ce6947aae485209439b65f2648eb0d4af"
MODEL_SHA256 = "7186c1fa052f161b43e49cb8dfbe21f65885f2e8ba2a15c1d75ed088a8da8ec5"
GITHUB_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases"


def state_dir() -> Path:
    return Path(os.environ.get("NL2SH_HOME", Path.home() / ".nl2sh")).expanduser()


def runtime_asset_name(system: str | None = None, machine: str | None = None) -> str:
    """Return the official llama.cpp CPU asset for the current platform."""
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    arm = machine in {"arm64", "aarch64"} or machine.startswith("arm")
    if system == "windows":
        return "llama-{tag}-bin-win-cpu-" + ("arm64.zip" if arm else "x64.zip")
    if system == "darwin":
        return "llama-{tag}-bin-macos-" + ("arm64.tar.gz" if arm else "x64.tar.gz")
    if system == "linux":
        return "llama-{tag}-bin-ubuntu-" + ("arm64.tar.gz" if arm else "x64.tar.gz")
    raise RuntimeError(f"unsupported operating system: {platform.system()}")


def _request_json(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "nl2sh-plus-setup/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _runtime_asset():
    pattern = runtime_asset_name()
    releases = _request_json(GITHUB_RELEASES + "?per_page=20")
    prerelease_match = None
    for release in releases:
        if release.get("draft"):
            continue
        tag = release.get("tag_name", "")
        expected_name = pattern.format(tag=tag)
        for asset in release.get("assets", []):
            if asset.get("name") == expected_name:
                digest = asset.get("digest", "")
                match = (release, asset, digest.removeprefix("sha256:") or None)
                if release.get("prerelease"):
                    prerelease_match = match
                else:
                    return match
    if prerelease_match:
        return prerelease_match
    raise RuntimeError(f"no CPU llama.cpp release asset found for {pattern}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str | None, label: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = _sha256(destination)
        if not expected_sha256 or actual == expected_sha256:
            print(f"Using existing {label}: {destination}")
            return destination
        destination.unlink()

    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()
    print(f"Downloading {label}...")
    request = urllib.request.Request(url, headers={"User-Agent": "nl2sh-plus-setup/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
        total = int(response.headers.get("Content-Length", "0") or 0)
        received = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            received += len(chunk)
            if total and (received == len(chunk) or received % (50 * 1024 * 1024) < len(chunk)):
                print(f"  {received / total:.0%}", flush=True)
    actual = _sha256(partial)
    if expected_sha256 and actual != expected_sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {label}: expected {expected_sha256}, got {actual}")
    partial.replace(destination)
    return destination


def _safe_target(root: Path, member_name: str) -> Path:
    root = root.resolve()
    target = (root / member_name).resolve()
    if os.path.commonpath([str(root), str(target)]) != str(root):
        raise RuntimeError(f"unsafe archive path: {member_name}")
    return target


def _extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise RuntimeError(f"symbolic links are not allowed in runtime archive: {member.filename}")
                _safe_target(destination, member.filename)
            bundle.extractall(destination)
        return
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if member.issym() or member.islnk():
                raise RuntimeError(f"symbolic links are not allowed in runtime archive: {member.name}")
            _safe_target(destination, member.name)
        bundle.extractall(destination)


def _find_server(root: Path) -> Path | None:
    names = {"llama-server.exe", "llama-server"}
    return next((path for path in root.rglob("*") if path.is_file() and path.name in names), None)


def install_model(root: Path | None = None) -> Path:
    root = root or state_dir()
    destination = root / "models" / MODEL_FILE
    url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_FILE}?download=true"
    return _download(url, destination, MODEL_SHA256, "nl2sh model")


def install_runtime(root: Path | None = None) -> tuple[Path, str]:
    root = root or state_dir()
    release, asset, digest = _runtime_asset()
    tag = release["tag_name"]
    archive_name = asset["name"]
    install_root = root / "runtime" / tag / Path(archive_name).stem.removesuffix(".tar")
    server = _find_server(install_root)
    if server:
        print(f"Using existing llama.cpp runtime: {server.parent}")
        return server.parent, tag

    downloads = root / "downloads"
    archive = downloads / archive_name
    _download(asset["browser_download_url"], archive, digest, f"llama.cpp {tag}")
    try:
        _extract(archive, install_root)
    finally:
        archive.unlink(missing_ok=True)
    server = _find_server(install_root)
    if not server:
        raise RuntimeError(f"llama-server was not found after extracting {archive_name}")
    return server.parent, tag
