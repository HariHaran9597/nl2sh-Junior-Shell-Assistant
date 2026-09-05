import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cli.nl2sh import _extract, MockBackend, _is_echo
def test_extract_code_fence(): assert _extract("```bash\nfind . -type f\n```")=="find . -type f"
def test_extract_dollar(): assert _extract("$ tar -czf a.tar.gz .")=="tar -czf a.tar.gz ."
def test_mock(): assert "tar" in MockBackend().generate("compress this folder")
def test_echo_single(): assert _is_echo("ls /tmp", ["tar -xzf a.tar.gz -C /tmp", "ls /tmp"])
def test_echo_multi(): assert _is_echo("tar -xzf a.tar.gz -C /tmp; ls /tmp", ["tar -xzf a.tar.gz -C /tmp", "ls /tmp"])
def test_not_echo(): assert not _is_echo("find /tmp -type f | head", ["tar -xzf a.tar.gz -C /tmp", "ls /tmp"])
def test_not_echo_empty_hist(): assert not _is_echo("ls -la", [])
