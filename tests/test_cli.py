import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from cli import nl2sh
from cli.nl2sh import _extract, MockBackend, _is_echo, OpenAICompatibleBackend
def test_extract_code_fence(): assert _extract("```bash\nfind . -type f\n```")=="find . -type f"
def test_extract_dollar(): assert _extract("$ tar -czf a.tar.gz .")=="tar -czf a.tar.gz ."
def test_mock(): assert "tar" in MockBackend().generate("compress this folder")
def test_echo_single(): assert _is_echo("ls /tmp", ["tar -xzf a.tar.gz -C /tmp", "ls /tmp"])
def test_echo_multi(): assert _is_echo("tar -xzf a.tar.gz -C /tmp; ls /tmp", ["tar -xzf a.tar.gz -C /tmp", "ls /tmp"])
def test_not_echo(): assert not _is_echo("find /tmp -type f | head", ["tar -xzf a.tar.gz -C /tmp", "ls /tmp"])
def test_not_echo_empty_hist(): assert not _is_echo("ls -la", [])

def test_education_explains_tool_flags_and_pipe():
    from src.education import explain_command
    result=explain_command("find . -type f -size +100M | head")
    assert "searches a directory tree" in result.lower()
    assert "regular files" in result
    assert "passes one command's output" in result

def test_hosted_backend_uses_bearer_key_without_persisting_it(monkeypatch):
    seen={}
    def fake_post(url,payload,timeout,api_key=None):
        seen.update(url=url,payload=payload,timeout=timeout,api_key=api_key)
        return {"choices":[{"message":{"content":"ls -la"}}]}
    monkeypatch.setattr(nl2sh,"_post",fake_post)
    backend=OpenAICompatibleBackend("https://api.example/v1","test-model","secret-test-key")
    assert backend.generate("list files")=="ls -la"
    assert seen["url"]=="https://api.example/v1/chat/completions"
    assert seen["api_key"]=="secret-test-key"
    assert "secret-test-key" not in json.dumps(seen["payload"])
    assert seen["payload"]["max_completion_tokens"]==128
    assert "max_tokens" not in seen["payload"]

def test_hosted_backend_local_http_roundtrip_sends_auth_header():
    seen={}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"]=self.path
            seen["auth"]=self.headers.get("Authorization")
            seen["body"]=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            payload=json.dumps({"choices":[{"message":{"content":"ls -la"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def log_message(self,*args): pass
    server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    worker=threading.Thread(target=server.serve_forever,daemon=True)
    worker.start()
    try:
        url=f"http://127.0.0.1:{server.server_port}/v1"
        backend=OpenAICompatibleBackend(url,"test-model","loopback-secret",timeout=3)
        assert backend.generate("list files")=="ls -la"
        assert seen["path"]=="/v1/chat/completions"
        assert seen["auth"]=="Bearer loopback-secret"
        assert seen["body"]["model"]=="test-model"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)

def test_api_backend_reads_key_from_environment_and_requires_model(monkeypatch,tmp_path):
    config=tmp_path/"config.json"
    monkeypatch.setenv("NL2SH_CONFIG",str(config))
    monkeypatch.setenv("GROQ_API_KEY","test-groq-key")
    config.write_text(json.dumps({"backend":"api","provider":"groq","model":"qwen/test"}))
    args=SimpleNamespace(mock=False,backend=None,provider=None,base_url=None,model=None,timeout=20)
    backend=nl2sh._build_backend(args)
    assert isinstance(backend,OpenAICompatibleBackend)
    assert backend.model=="qwen/test"
    assert backend.api_key=="test-groq-key"
    assert "test-groq-key" not in config.read_text()

def test_setup_keeps_hosted_model_id_and_does_not_save_key(monkeypatch,tmp_path,capsys):
    config=tmp_path/"config.json"
    monkeypatch.setenv("NL2SH_CONFIG",str(config))
    assert nl2sh._setup(["--provider","groq","--model","qwen/qwen3-8b"])==0
    saved=json.loads(config.read_text())
    assert saved["model"]=="qwen/qwen3-8b"
    assert saved["provider"]=="groq"
    assert "API keys are not stored" in capsys.readouterr().out

def test_init_bash_prints_review_before_enter_integration(capsys):
    assert nl2sh._init(["bash"])==0
    output=capsys.readouterr().out
    assert "READLINE_LINE" in output
    assert "press Enter yourself" in output

def test_generated_suggestion_is_not_executed_by_default(monkeypatch,capsys):
    monkeypatch.setattr(nl2sh,"record",lambda *args: None)
    assert nl2sh.process("find files bigger than 100 MB",MockBackend(),execute=False,with_context=False)==0
    output=capsys.readouterr().out
    assert "find . -size +100M" in output
    assert "What it does:" in output
    assert "Run?" not in output
    assert "[MED]" in output

def test_custom_hosted_endpoint_must_be_configured(monkeypatch,tmp_path):
    config=tmp_path/"config.json"
    monkeypatch.setenv("NL2SH_CONFIG",str(config))
    monkeypatch.setenv("NL2SH_API_KEY","test-key")
    config.write_text(json.dumps({"backend":"api","provider":"custom","model":"my-model"}))
    args=SimpleNamespace(mock=False,backend=None,provider=None,base_url=None,model=None,timeout=20)
    try:
        nl2sh._build_backend(args)
    except SystemExit as exc:
        assert "requires --base-url" in str(exc)
    else:
        raise AssertionError("custom API endpoint without a URL should fail clearly")

def test_setup_switches_from_hosted_api_to_local_gguf(monkeypatch,tmp_path,capsys):
    config=tmp_path/"config.json"
    model=tmp_path/"new-model.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setenv("NL2SH_CONFIG",str(config))
    config.write_text(json.dumps({"backend":"api","provider":"groq","model":"provider/model","base_url":"https://api.groq.example/v1"}))
    monkeypatch.setattr("src.local_setup.install_runtime",lambda: (tmp_path/"runtime","test-runtime"))
    assert nl2sh._setup(["--backend","llama-cpp","--model",str(model)])==0
    saved=json.loads(config.read_text())
    assert saved["backend"]=="llama-cpp"
    assert saved["model"]==str(model)
    assert "provider" not in saved
    assert saved["base_url"]=="http://127.0.0.1:8080/v1"
    assert "test-runtime" in capsys.readouterr().out

def test_setup_rejects_missing_local_model_file(monkeypatch,tmp_path,capsys):
    monkeypatch.setenv("NL2SH_CONFIG",str(tmp_path/"config.json"))
    assert nl2sh._setup(["--backend","llama-cpp","--model",str(tmp_path/"missing.gguf")])==2
    assert "does not exist" in capsys.readouterr().err

def test_api_doctor_hides_key_and_returns_configuration_status(monkeypatch,tmp_path,capsys):
    config=tmp_path/"config.json"
    monkeypatch.setenv("NL2SH_CONFIG",str(config))
    monkeypatch.setenv("GROQ_API_KEY","test-only-secret")
    config.write_text(json.dumps({"backend":"api","provider":"groq","model":"qwen/test"}))
    assert nl2sh._doctor([])==0
    output=capsys.readouterr().out
    assert "Configuration looks complete" in output
    assert "test-only-secret" not in output
    assert "value hidden" in output

def test_local_doctor_fails_when_model_missing(monkeypatch,tmp_path,capsys):
    monkeypatch.setenv("NL2SH_CONFIG",str(tmp_path/"config.json"))
    assert nl2sh._doctor([])==1
    assert "no model" in capsys.readouterr().out
